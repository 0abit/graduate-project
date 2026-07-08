import json
import re
import sys
import unicodedata
from collections import OrderedDict
from datetime import datetime
from pathlib import Path

from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class LiteratureChunker:

    PAGE_MARKER_RE = re.compile(r"<!--\s*OCR_PAGE:\s*(\d+)\s*-->", re.IGNORECASE)
    HTML_COMMENT_RE = re.compile(r"<!--.*?-->", re.DOTALL)

    def __init__(self):
        self.client = MongoClient(config.MONGO_URI)
        self.collection = self.client[config.DB_NAME][config.COLLECTION_NAME_V2]
        self.toc_items = []
        self.lesson_metadata = {}

    def process(self, md_filename, book_id):
        print(f"📦 [PHASE 2] Đang chunk theo TOC từ OCR Markdown: {md_filename}")
        md_path = Path(config.MD_DIR) / md_filename
        if not md_path.exists():
            raise FileNotFoundError(f"Không tìm thấy Markdown OCR: {md_path}")

        self.toc_items = self._load_toc_items(book_id)
        full_text = md_path.read_text(encoding="utf-8")
        pages, has_page_markers = self._read_ocr_pages(full_text)

        if has_page_markers:
            self.lesson_metadata = self._load_or_extract_lesson_metadata(book_id, pages)
            chunks_to_save = self._build_toc_chunks(pages, book_id)
        else:
            print("⚠️ File Markdown chưa có OCR_PAGE marker; dùng fallback chunk theo heading cũ.")
            self.lesson_metadata = {}
            chunks_to_save = self._legacy_heading_chunks(full_text, book_id)

        if not chunks_to_save:
            print("⚠️ Cảnh báo: Không tìm thấy chunk nào!")
            return

        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.collection.update_many(
            {"metadata.book_id": book_id},
            {"$set": {"metadata.active": False, "metadata.replaced_by_run_id": run_id}},
        )

        total = len(chunks_to_save)
        try:
            for index, chunk in enumerate(chunks_to_save, start=1):
                if self._stop_requested():
                    raise KeyboardInterrupt

                chunk_id = chunk["_id"]
                chunk["metadata"]["active"] = True
                chunk["metadata"]["ingest_run_id"] = run_id
                chunk["metadata"]["phase2_upserted_at"] = datetime.now().isoformat()

                doc_to_set = dict(chunk)
                doc_to_set.pop("_id", None)
                self.collection.update_one({"_id": chunk_id}, {"$set": doc_to_set}, upsert=True)
                self._write_phase_checkpoint(book_id, "running", index, total, chunk_id, run_id)
        except KeyboardInterrupt:
            print("🛑 Phase 2 đã dừng an toàn. Chạy lại phase2.py để tiếp tục/upsert lại các chunk còn thiếu.")
            raise

        deleted = self.collection.delete_many({"metadata.book_id": book_id, "metadata.active": False})
        self._write_phase_checkpoint(book_id, "complete", total, total, None, run_id)
        print(f"✅ Đã upsert {total} chunk mới vào MongoDB và xóa {deleted.deleted_count} chunk cũ.")

    def _load_toc_items(self, book_id):
        if hasattr(config, "get_toc_mapping"):
            toc = config.get_toc_mapping(book_id)
        else:
            toc = getattr(config, "TOC_MAPPING", {})
        items = []
        for lesson_id, item in toc.items():
            try:
                items.append((lesson_id, int(item["pdf_start"]), int(item["pdf_end"]), item["title"]))
            except (KeyError, TypeError, ValueError):
                print(f"⚠️ Bỏ qua TOC_MAPPING không hợp lệ: {lesson_id} -> {item}")
        return sorted(items, key=lambda item: item[1])

    def _read_ocr_pages(self, full_text):
        markers = list(self.PAGE_MARKER_RE.finditer(full_text))
        if not markers:
            return [], False

        pages = []
        for index, marker in enumerate(markers):
            start = marker.end()
            end = markers[index + 1].start() if index + 1 < len(markers) else len(full_text)
            page_text = full_text[start:end]
            page_text = self.HTML_COMMENT_RE.sub("", page_text).strip()
            pages.append({"page_number": int(marker.group(1)), "text": page_text})
        return pages, True

    def _load_or_extract_lesson_metadata(self, book_id, pages):
        metadata_path = Path(config.METADATA_DIR) / f"{book_id}_lessons.json"
        generated = self._extract_lesson_metadata(book_id, pages)
        existing = self._read_lesson_metadata(metadata_path)

        merged = OrderedDict()
        for lesson_id, lesson_data in generated.items():
            merged[lesson_id] = self._merge_lesson_metadata(lesson_data, existing.get(lesson_id, {}))

        for lesson_id, lesson_data in existing.items():
            if lesson_id not in merged:
                merged[lesson_id] = lesson_data

        metadata_path.parent.mkdir(parents=True, exist_ok=True)
        metadata_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")

        found = sum(1 for item in merged.values() if item.get("learning_objectives"))
        print(f"🧭 Đã cập nhật metadata bài học: {metadata_path} ({found}/{len(generated)} bài có learning_objectives)")
        return merged

    def _read_lesson_metadata(self, metadata_path):
        if not metadata_path.exists():
            return {}
        try:
            return json.loads(metadata_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            print(f"⚠️ Không đọc được metadata JSON, sẽ tạo lại từ OCR: {metadata_path} ({exc})")
            return {}

    def _merge_lesson_metadata(self, generated, existing):
        if not existing:
            return generated

        merged = dict(generated)
        merged.update(existing)

        generated_objectives = generated.get("learning_objectives") or []
        existing_objectives = existing.get("learning_objectives") or []
        reviewed = bool(existing.get("learning_objectives_reviewed"))
        force_reextract = getattr(config, "PHASE2_FORCE_METADATA_REEXTRACT", True)
        existing_source = str(existing.get("objective_source", ""))

        should_replace_objectives = (
            generated_objectives
            and not reviewed
            and (
                force_reextract
                or not existing_objectives
                or existing_source.startswith("ocr")
            )
        )
        if should_replace_objectives:
            merged["learning_objectives"] = generated_objectives
            merged["objective_source"] = generated.get("objective_source", "ocr:auto")
            merged["learning_objectives_reviewed"] = False

        return merged

    def _extract_lesson_metadata(self, book_id, pages):
        page_lookup = {page["page_number"]: page for page in pages}
        scan_pages = max(1, getattr(config, "OBJECTIVE_SCAN_PAGES", 2))
        metadata = OrderedDict()

        for lesson_id, pdf_start, pdf_end, title in self.toc_items:
            page_end = min(pdf_end, pdf_start + scan_pages - 1)
            lesson_text = "\n".join(
                page_lookup[page_number]["text"]
                for page_number in range(pdf_start, page_end + 1)
                if page_number in page_lookup
            )
            learning_objectives = self._extract_learning_objectives(lesson_text)
            metadata[lesson_id] = {
                "book_id": book_id,
                "grade": self._grade_from_book_id(book_id),
                "lesson_id": lesson_id,
                "lesson_title": title,
                "lesson_page_start": pdf_start,
                "lesson_page_end": pdf_end,
                "source_level": "textbook",
                "learning_objectives": learning_objectives,
                "objective_source": "ocr:auto" if learning_objectives else "ocr:not_found",
                "learning_objectives_reviewed": False,
            }

        return metadata

    def _extract_learning_objectives(self, text):
        lines = [self._clean_title(line) for line in text.splitlines()]
        lines = [line for line in lines if line]
        start_index = self._find_objective_start(lines)
        if start_index is None:
            return []

        objectives = []
        current = []

        for line in lines[start_index:]:
            if self._is_objective_stop_line(line, has_objective=bool(objectives or current)):
                break

            is_bullet = self._is_objective_bullet(line)
            clean = self._strip_objective_bullet(line)
            if not clean:
                continue

            if is_bullet:
                self._push_objective(objectives, current)
                current = [clean]
            elif current:
                current.append(clean)
            elif self._looks_like_objective(clean):
                current = [clean]

        self._push_objective(objectives, current)
        return self._dedupe_objectives(objectives)

    def _find_objective_start(self, lines):
        for index, line in enumerate(lines):
            normalized = self._normalize_for_match(line)
            if "yeu cau can dat" in normalized:
                return index + 1
        return None

    def _is_objective_stop_line(self, line, has_objective):
        clean = self._clean_title(line)
        normalized = self._normalize_for_match(clean)

        if has_objective and clean.endswith("?"):
            return True
        if normalized.startswith(("muc tieu", "mo dau", "khoi dong", "luyen tap", "van dung")):
            return True
        if re.match(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x)\s+", normalized):
            return True
        if re.match(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x)\.\s+", self._strip_accents(clean).lower()):
            return True
        return has_objective and self._is_section_heading(clean)

    def _is_objective_bullet(self, line):
        return bool(re.match(r"^[\-–—•·*+▪■●◦o]\s*", line.strip()))

    def _strip_objective_bullet(self, line):
        clean = re.sub(r"^[\-–—•·*+▪■●◦o]\s*", "", line.strip())
        clean = re.sub(r"\s+", " ", clean).strip(" ;,.\t")
        return clean

    def _looks_like_objective(self, line):
        normalized = self._normalize_for_match(line)
        prefixes = (
            "dua vao", "neu duoc", "phan tich duoc", "trinh bay duoc", "mo ta duoc",
            "giai thich duoc", "xac dinh duoc", "thuc hien duoc", "van dung duoc",
            "nhan biet duoc", "ke duoc", "lay duoc", "lap duoc", "de xuat duoc",
        )
        return normalized.startswith(prefixes) and len(line) > 12

    def _push_objective(self, objectives, current):
        text = " ".join(part.strip() for part in current if part.strip())
        text = re.sub(r"\s+", " ", text).strip(" ;,.\t")
        if len(text) >= 12:
            objectives.append(text + ("." if not text.endswith((".", "?", "!")) else ""))

    def _dedupe_objectives(self, objectives):
        seen = set()
        deduped = []
        for objective in objectives:
            key = self._normalize_for_match(objective)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(objective)
        return deduped

    def _normalize_for_match(self, value):
        value = self._strip_accents(value).lower()
        return re.sub(r"[^a-z0-9]+", " ", value).strip()

    def _grade_from_book_id(self, book_id):
        match = re.search(r"(\d+)", book_id)
        return int(match.group(1)) if match else None

    def _chunk_uid(self, book_id, lesson_id, chunk_index):
        safe_lesson_id = re.sub(r"[^A-Za-z0-9_\-]+", "_", str(lesson_id or "unknown")).strip("_")
        return f"{book_id}:{safe_lesson_id}:{int(chunk_index):04d}"

    def _write_phase_checkpoint(self, book_id, status, current_chunk, total_chunks, chunk_id, run_id):
        checkpoint_path = Path(config.CHECKPOINT_DIR) / f"{book_id}_phase2.json"
        payload = {
            "phase": "phase2",
            "book_id": book_id,
            "status": status,
            "current_chunk": current_chunk,
            "total_chunks": total_chunks,
            "chunk_id": chunk_id,
            "run_id": run_id,
            "updated_at": datetime.now().isoformat(),
        }
        self._atomic_write_text(checkpoint_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _atomic_write_text(self, path, content):
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
    def _stop_requested(self):
        return Path(config.STOP_FILE).exists()
    def _build_toc_chunks(self, pages, book_id):
        grouped_pages, skipped_pages = self._group_pages_by_lesson(pages)
        if skipped_pages:
            print(f"ℹ️ Bỏ qua {len(skipped_pages)} trang ngoài TOC: {skipped_pages[:10]}{'...' if len(skipped_pages) > 10 else ''}")

        chunks = []
        global_index = 0
        for lesson_id, payload in grouped_pages.items():
            lesson = payload["lesson"]
            lesson_pages = payload["pages"]
            lesson_chunks = self._chunks_for_lesson(book_id, lesson_id, lesson, lesson_pages)
            for chunk in lesson_chunks:
                global_index += 1
                chunk["metadata"]["global_chunk_index"] = global_index
                chunks.append(chunk)
        return chunks

    def _group_pages_by_lesson(self, pages):
        grouped = OrderedDict()
        skipped_pages = []

        for page in pages:
            lesson = self._lesson_for_page(page["page_number"])
            if lesson is None:
                if not config.INCLUDE_OUT_OF_TOC_PAGES:
                    skipped_pages.append(page["page_number"])
                    continue
                lesson = {
                    "lesson_id": "Ngoai_TOC",
                    "pdf_start": page["page_number"],
                    "pdf_end": page["page_number"],
                    "title": "Phần ngoài mục lục",
                }

            lesson_id = lesson["lesson_id"]
            if lesson_id not in grouped:
                grouped[lesson_id] = {"lesson": lesson, "pages": []}
            grouped[lesson_id]["pages"].append(page)

        return grouped, skipped_pages

    def _lesson_for_page(self, page_number):
        for lesson_id, pdf_start, pdf_end, title in self.toc_items:
            if pdf_start <= page_number <= pdf_end:
                return {
                    "lesson_id": lesson_id,
                    "pdf_start": pdf_start,
                    "pdf_end": pdf_end,
                    "title": title,
                }
        return None

    def _chunks_for_lesson(self, book_id, lesson_id, lesson, pages):
        blocks = self._section_blocks(lesson["title"], pages)
        chunks = []
        lesson_chunk_index = 0
        lesson_meta = self.lesson_metadata.get(lesson_id, {})

        for block in blocks:
            for piece in self._split_text(block["text"], config.CHUNK_MAX_CHARS):
                piece = piece.strip()
                if not piece:
                    continue
                lesson_chunk_index += 1
                source_pages = sorted(block["pages"])
                section_title = block["section_title"] or lesson["title"]
                chunk_uid = self._chunk_uid(book_id, lesson_id, lesson_chunk_index)
                chunks.append({
                    "_id": chunk_uid,
                    "metadata": {
                        "book_id": book_id,
                        "chunk_uid": chunk_uid,
                        "grade": lesson_meta.get("grade", self._grade_from_book_id(book_id)),
                        "lesson_id": lesson_id,
                        "lesson_title": lesson["title"],
                        "section_title": section_title,
                        "lesson_page_start": lesson["pdf_start"],
                        "lesson_page_end": lesson["pdf_end"],
                        "source_pages": source_pages,
                        "source_level": lesson_meta.get("source_level", "textbook"),
                        "learning_objectives": lesson_meta.get("learning_objectives", []),
                        "learning_objectives_reviewed": lesson_meta.get("learning_objectives_reviewed", False),
                        "chunk_index": lesson_chunk_index,
                        "parser": "paddleocr",
                        "api_refined": False,
                    },
                    "markdown_content": self._format_chunk_content(lesson["title"], section_title, source_pages, piece),
                })
        return chunks

    def _section_blocks(self, lesson_title, pages):
        blocks = []
        current_section = lesson_title
        current_lines = []
        current_pages = set()

        def flush():
            nonlocal current_lines, current_pages
            text = "\n".join(current_lines).strip()
            if text:
                blocks.append({
                    "section_title": current_section,
                    "text": text,
                    "pages": set(current_pages),
                })
            current_lines = []
            current_pages = set()

        for page in pages:
            page_number = page["page_number"]
            for raw_line in page.get("text", "").splitlines():
                line = raw_line.strip()
                if not line:
                    continue
                if self._is_section_heading(line):
                    flush()
                    current_section = self._clean_title(line)
                current_lines.append(line)
                current_pages.add(page_number)
            current_lines.append("")

        flush()
        return blocks

    def _split_text(self, text, max_chars):
        if len(text) <= max_chars:
            return [text]

        parts = []
        buffer = []
        buffer_len = 0
        for line in text.splitlines():
            line_len = len(line) + 1
            if buffer and buffer_len + line_len > max_chars:
                parts.append("\n".join(buffer))
                buffer = []
                buffer_len = 0
            buffer.append(line)
            buffer_len += line_len
        if buffer:
            parts.append("\n".join(buffer))
        return parts

    def _format_chunk_content(self, lesson_title, section_title, source_pages, text):
        pages_label = self._pages_label(source_pages)
        header = [f"# {lesson_title}"]
        if section_title and section_title != lesson_title:
            header.append(f"## {section_title}")
        header.append(f"<!-- source_pages: {pages_label} -->")
        return "\n\n".join(header + [text.strip()])

    def _pages_label(self, pages):
        if not pages:
            return ""
        if len(pages) == 1:
            return str(pages[0])
        return f"{pages[0]}-{pages[-1]}"

    def _is_section_heading(self, line):
        clean = self._clean_title(line)
        if len(clean) < 3 or len(clean) > 140 or clean.endswith("?"):
            return False

        normalized = self._strip_accents(clean).lower()
        if normalized.startswith(("hinh ", "bang ", "bieu do ", "so do ")):
            return False

        if re.match(r"^(bai|chuong)\s*\d+\b", normalized):
            return True
        if re.match(r"^(i|ii|iii|iv|v|vi|vii|viii|ix|x|xi|xii)\.\s+", normalized):
            return True
        if re.match(r"^\d+[\.)]\s+", normalized) and len(clean) <= 90:
            return True

        letters = [char for char in clean if char.isalpha()]
        if len(letters) < 5:
            return False
        upper_ratio = sum(1 for char in letters if char.isupper()) / len(letters)
        word_count = len(clean.split())
        return upper_ratio >= 0.65 and 2 <= word_count <= 16

    def _clean_title(self, line):
        return re.sub(r"\s+", " ", line.strip(" #:-\t")).strip()

    def _strip_accents(self, value):
        value = value.replace("đ", "d").replace("Đ", "D")
        normalized = unicodedata.normalize("NFD", value)
        return "".join(char for char in normalized if unicodedata.category(char) != "Mn")

    def _legacy_heading_chunks(self, full_text, book_id):
        pattern = re.compile(r"(^#{1,3}\s+.*?)(?=^#{1,3}\s+|\Z)", re.MULTILINE | re.DOTALL)
        all_chunks = pattern.findall(full_text)
        chunks_to_save = []
        current_lesson = "Phần Mở Đầu"

        for index, chunk_content in enumerate(all_chunks, start=1):
            chunk_content = chunk_content.strip()
            if not chunk_content:
                continue

            section_title_match = re.match(r"^#{1,3}\s+(.*)", chunk_content)
            section_title = section_title_match.group(1).strip() if section_title_match else "Nội dung"
            normalized = self._strip_accents(section_title).lower()
            if "bai" in normalized or "chuong" in normalized or "phan" in normalized:
                current_lesson = section_title

            legacy_chunk_uid = self._chunk_uid(book_id, "legacy", index)
            chunks_to_save.append({
                "_id": legacy_chunk_uid,
                "metadata": {
                    "book_id": book_id,
                    "chunk_uid": legacy_chunk_uid,
                    "grade": self._grade_from_book_id(book_id),
                    "lesson_id": None,
                    "lesson_title": current_lesson,
                    "section_title": section_title,
                    "source_pages": [],
                    "source_level": "textbook",
                    "learning_objectives": [],
                    "chunk_index": index,
                    "parser": "legacy_markdown",
                    "api_refined": False,
                },
                "markdown_content": chunk_content,
            })
        return chunks_to_save


if __name__ == "__main__":
    chunker = LiteratureChunker()
    chunker.process(f"{config.CURRENT_BOOK}.md", config.CURRENT_BOOK)










