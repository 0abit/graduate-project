import json
import os
import statistics
import sys
import re
import io
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class PDFConverter:
    """Convert PDF textbooks to page-aware OCR markdown.

    Default mode is PaddleOCR for text-line boxes plus VietOCR for Vietnamese
    recognition. The output contract is unchanged for the later phases.
    """

    def __init__(self, *_unused, lang=None, ocr_version=None, device=None, ocr_mode=None):
        self.lang = lang or config.PADDLE_OCR_LANG
        self.ocr_version = ocr_version or config.PADDLE_OCR_VERSION
        self.device = device or config.PADDLE_DEVICE
        self.ocr_mode = (ocr_mode or getattr(config, "PHASE1_OCR_MODE", "paddle_vietocr")).lower()
        self._engine = None
        self._api_version = None
        self._vietocr = None

    def _configure_paddle_runtime(self):
        if not config.PADDLE_ENABLE_MKLDNN:
            os.environ["FLAGS_use_mkldnn"] = "0"
            os.environ["FLAGS_use_onednn"] = "0"
            os.environ["PADDLE_PDX_ENABLE_MKLDNN_BYDEFAULT"] = "0"
            os.environ.setdefault("FLAGS_enable_pir_api", "0")
        os.environ.setdefault("PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK", "True")

    def _build_engine(self):
        if self._engine is not None:
            return self._engine, self._api_version

        self._configure_paddle_runtime()

        try:
            from paddleocr import PaddleOCR
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài PaddleOCR. Hãy cài paddlepaddle + paddleocr trước khi chạy Phase 1."
            ) from exc

        v3_kwargs = {
            "lang": self.lang,
            "ocr_version": self.ocr_version,
            "use_doc_orientation_classify": False,
            "use_doc_unwarping": False,
            "use_textline_orientation": config.PADDLE_USE_TEXTLINE_ORIENTATION,
        }
        if self.device:
            v3_kwargs["device"] = self.device

        try:
            self._engine = PaddleOCR(**v3_kwargs)
            self._api_version = "paddleocr_v3" if hasattr(self._engine, "predict") else "paddleocr_v2"
            return self._engine, self._api_version
        except (TypeError, ValueError, AssertionError):
            v2_kwargs = {
                "use_angle_cls": config.PADDLE_USE_TEXTLINE_ORIENTATION,
                "lang": self.lang,
                "show_log": False,
            }
            if self.ocr_version:
                v2_kwargs["ocr_version"] = self.ocr_version
            if self.device:
                v2_kwargs["use_gpu"] = self.device.lower().startswith(("gpu", "cuda"))
            self._engine = PaddleOCR(**v2_kwargs)
            self._api_version = "paddleocr_v2"
            return self._engine, self._api_version

    def _build_vietocr(self):
        if self._vietocr is not None:
            return self._vietocr

        try:
            from vietocr.tool.config import Cfg
            from vietocr.tool.predictor import Predictor
        except ImportError as exc:
            raise RuntimeError(
                "Chưa cài VietOCR. Hãy cài torch + vietocr, hoặc đặt PHASE1_OCR_MODE=paddle."
            ) from exc

        viet_config = Cfg.load_config_from_name(config.VIETOCR_CONFIG)
        viet_config["device"] = self._vietocr_device()
        if "predictor" in viet_config:
            viet_config["predictor"]["beamsearch"] = config.VIETOCR_BEAMSEARCH
        self._vietocr = Predictor(viet_config)
        return self._vietocr

    def _vietocr_device(self):
        device = str(getattr(config, "VIETOCR_DEVICE", "cpu") or "cpu").lower()
        if device.startswith("gpu") or device.startswith("cuda"):
            return "cuda:0"
        return "cpu"

    def _use_vietocr(self):
        return self.ocr_mode in {"paddle_vietocr", "vietocr", "paddle+vietocr"}

    def _ocr_engine_name(self):
        return "paddleocr+vietocr" if self._use_vietocr() else "paddleocr"

    def convert(self, pdf_filename):
        print(f"🚀 [PHASE 1] Đang OCR PDF bằng {self._ocr_engine_name()}: {pdf_filename}")
        pdf_path = Path(config.PDF_DIR) / pdf_filename
        if not pdf_path.exists():
            raise FileNotFoundError(f"Không tìm thấy PDF: {pdf_path}")

        book_id = pdf_path.stem
        md_path = Path(config.MD_DIR) / f"{book_id}.md"
        json_path = Path(config.OCR_JSON_DIR) / f"{book_id}.pages.json"

        try:
            pages = self._ocr_pdf(pdf_path, json_path, md_path, book_id)
        except KeyboardInterrupt:
            print("🛑 Phase 1 đã dừng an toàn. Chạy lại phase1.py để tiếp tục từ checkpoint.")
            raise

        if not pages:
            raise RuntimeError("OCR không trả về nội dung nào từ PDF.")

        self._write_markdown(md_path, book_id, pages)
        self._write_json(json_path, book_id, pages, status="complete", total_pages=len(pages))
        self._write_phase_checkpoint(book_id, "complete", len(pages), len(pages), json_path, md_path)

        print(f"✅ Đã lưu Markdown OCR theo trang: {md_path}")
        print(f"✅ Đã lưu JSON OCR/checkpoint: {json_path}")
        return str(md_path)

    def _ocr_pdf(self, pdf_path, json_path, md_path, book_id):
        # Import Torch/VietOCR before PaddleOCR to avoid Windows DLL conflicts.
        if self._use_vietocr():
            self._build_vietocr()
        engine, api_version = self._build_engine()

        image_paths = self._render_pdf_pages(pdf_path)
        total_pages = len(image_paths)

        completed = {}
        if not config.PHASE1_FORCE_REOCR:
            for page in self._load_pages_from_json(json_path):
                page_number = page.get("page_number")
                if isinstance(page_number, int) and page.get("ocr_engine") == self._ocr_engine_name():
                    completed[page_number] = page

        if completed:
            print(f"↩️ Phase 1 resume: đã có {len(completed)}/{total_pages} trang trong checkpoint.")

        for page_number, image_path in image_paths:
            if self._stop_requested():
                raise KeyboardInterrupt

            if page_number in completed and not config.PHASE1_FORCE_REOCR:
                print(f"  ⏭️ Bỏ qua trang {page_number}/{total_pages} (đã OCR bằng {self._ocr_engine_name()}).")
                continue

            print(f"  📄 OCR trang {page_number}/{total_pages}: {image_path.name}")
            page = self._ocr_image(engine, api_version, image_path, page_number)

            # --- V2: EXTRACT IMAGES ---
            page = self._extract_and_merge_images_from_pdf(pdf_path, page_number - 1, page, book_id)
            # --------------------------

            completed[page_number] = page

            pages = self._sorted_pages(completed.values())
            self._write_json(json_path, book_id, pages, status="running", total_pages=total_pages)
            self._write_markdown(md_path, book_id, pages)
            self._write_phase_checkpoint(book_id, "running", page_number, total_pages, json_path, md_path)

        return self._sorted_pages(completed.values())

    def _ocr_image(self, engine, api_version, image_path, page_number):
        if api_version == "paddleocr_v3" and hasattr(engine, "predict"):
            raw = engine.predict(str(image_path))
            if isinstance(raw, (list, tuple)):
                results = list(raw)
            else:
                try:
                    results = list(raw)
                except TypeError:
                    results = [raw]
            page = self._normalize_v3_result(results[0], page_number) if results else self._page_record(page_number, [])
        else:
            result = engine.ocr(str(image_path), cls=config.PADDLE_USE_TEXTLINE_ORIENTATION)
            lines = self._normalize_v2_lines(result)
            page = self._page_record(page_number, lines, page_index=page_number - 1)

        if self._use_vietocr() and page.get("lines"):
            page = self._recognize_page_with_vietocr(image_path, page, page_number)

        page["page_number"] = page_number
        page["page_index"] = page_number - 1
        page["ocr_engine"] = self._ocr_engine_name()
        return page

    def _recognize_page_with_vietocr(self, image_path, page, page_number):
        try:
            from PIL import Image
        except ImportError as exc:
            raise RuntimeError("VietOCR cần Pillow. Hãy cài Pillow hoặc cài lại vietocr.") from exc

        recognizer = self._build_vietocr()
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            viet_lines = []
            for index, line in enumerate(page.get("lines", []), start=1):
                bbox = line.get("bbox")
                if not bbox:
                    viet_lines.append(line)
                    continue

                crop = self._crop_line(image, bbox)
                if crop is None:
                    viet_lines.append(line)
                    continue

                try:
                    text = str(recognizer.predict(crop)).strip()
                except Exception as exc:
                    print(f"    ⚠️ VietOCR lỗi ở trang {page_number}, dòng {index}; dùng text PaddleOCR. ({exc})")
                    text = line.get("text", "")

                if not text:
                    text = line.get("text", "")

                new_line = dict(line)
                new_line["text"] = text
                if getattr(config, "PHASE1_KEEP_PADDLE_TEXT", False):
                    new_line["paddle_text"] = line.get("text", "")
                new_line["recognizer"] = "vietocr"
                viet_lines.append(new_line)

        return self._page_record(page_number, viet_lines, page_index=page.get("page_index"))

    def _crop_line(self, image, bbox):
        rect = self._bbox_rect(bbox)
        if not rect:
            return None

        left, top, right, bottom = rect
        padding = max(0, int(getattr(config, "VIETOCR_CROP_PADDING", 4)))
        left = max(0, int(left) - padding)
        top = max(0, int(top) - padding)
        right = min(image.width, int(right) + padding)
        bottom = min(image.height, int(bottom) + padding)

        if right <= left or bottom <= top:
            return None
        return image.crop((left, top, right, bottom))

    def _bbox_rect(self, bbox):
        try:
            if isinstance(bbox, dict):
                values = [bbox.get("x1"), bbox.get("y1"), bbox.get("x2"), bbox.get("y2")]
                if all(value is not None for value in values):
                    return tuple(float(value) for value in values)
                return None

            if isinstance(bbox, (list, tuple)) and len(bbox) == 4 and all(isinstance(value, (int, float)) for value in bbox):
                x1, y1, x2, y2 = bbox
                return (float(x1), float(y1), float(x2), float(y2))

            points = []
            for point in bbox:
                if isinstance(point, (list, tuple)) and len(point) >= 2:
                    points.append((float(point[0]), float(point[1])))
            if not points:
                return None
            xs = [point[0] for point in points]
            ys = [point[1] for point in points]
            return (min(xs), min(ys), max(xs), max(ys))
        except (TypeError, ValueError):
            return None

    def _extract_and_merge_images_from_pdf(self, pdf_path, page_idx, page_data, book_id):
        try:
            import fitz
        except ImportError:
            return page_data

        page_number = page_idx + 1
        book_extracted_image_dir = Path(config.EXTRACTED_IMAGE_DIR) / book_id
        
        use_cloudinary = bool(getattr(config, "CLOUDINARY_URL", None))
        if use_cloudinary:
            try:
                import cloudinary.uploader
                match = re.match(r"cloudinary://([^:]+):([^@]+)@(.+)", config.CLOUDINARY_URL)
                if match:
                    api_key, api_secret, cloud_name = match.groups()
                    cloudinary.config(cloud_name=cloud_name, api_key=api_key, api_secret=api_secret, secure=True)
                else:
                    use_cloudinary = False
            except ImportError:
                use_cloudinary = False
        else:
            book_extracted_image_dir.mkdir(parents=True, exist_ok=True)

        with fitz.open(str(pdf_path)) as doc:
            pdf_page = doc[page_idx]
            page_dict = pdf_page.get_text("dict")
            
            extracted_images = []
            img_idx = 0
            for block in page_dict.get("blocks", []):
                if block.get("type") == 1:  # Image block
                    img_bytes = block.get("image")
                    if img_bytes:
                        img_idx += 1
                        img_filename = f"page_{page_number:04d}_img_{img_idx}.png"
                        
                        if use_cloudinary:
                            # Upload to Cloudinary directly from memory
                            res = cloudinary.uploader.upload(
                                io.BytesIO(img_bytes), 
                                folder=f"bioai_rag/{book_id}", 
                                public_id=Path(img_filename).stem
                            )
                            md_text = f"![Hình ảnh minh hoạ]({res['secure_url']})"
                        else:
                            # Fallback: Save to local disk
                            img_path = book_extracted_image_dir / img_filename
                            with open(img_path, "wb") as f:
                                f.write(img_bytes)
                            md_text = f"![Hình ảnh minh hoạ](/static/extracted_images/{book_id}/{img_filename})"
                        
                        zoom = config.PADDLE_PDF_DPI / 72
                        x0, y0, x1, y1 = block["bbox"]
                        scaled_bbox = [x0 * zoom, y0 * zoom, x1 * zoom, y1 * zoom]
                        
                        extracted_images.append({
                            "text": md_text,
                            "score": 1.0,
                            "bbox": scaled_bbox,
                            "detector": "fitz_image"
                        })
            
            if extracted_images:
                all_lines = page_data.get("lines", []) + extracted_images
                all_lines.sort(key=lambda item: self._bbox_sort_key(item.get("bbox")))
                page_data["lines"] = all_lines
                page_data["text"] = "\n".join(line["text"] for line in all_lines if line.get("text"))
        
        return page_data

    def _render_pdf_pages(self, pdf_path):
        try:
            import fitz
        except ImportError as exc:
            raise RuntimeError(
                "Phase 1 checkpoint cần PyMuPDF để render PDF theo từng trang. Hãy cài pymupdf."
            ) from exc

        book_image_dir = Path(config.OCR_IMAGE_DIR) / pdf_path.stem
        self._ensure_directory(book_image_dir)

        zoom = config.PADDLE_PDF_DPI / 72
        matrix = fitz.Matrix(zoom, zoom)
        image_paths = []
        with fitz.open(str(pdf_path)) as doc:
            for page_idx in range(doc.page_count):
                page_number = page_idx + 1
                image_path = book_image_dir / f"page_{page_number:04d}.png"
                if not image_path.exists():
                    pix = doc.load_page(page_idx).get_pixmap(matrix=matrix, alpha=False)
                    pix.save(str(image_path))
                image_paths.append((page_number, image_path))
        return image_paths

    def _ensure_directory(self, directory):
        directory = Path(directory)
        try:
            if directory.exists() and not directory.is_dir():
                raise RuntimeError(f"Đường dẫn đang là file, không phải thư mục: {directory}")
            directory.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise RuntimeError(
                f"Không tạo được thư mục: {directory}. Hãy kiểm tra quyền ghi hoặc tạo thủ công rồi chạy lại."
            ) from exc

    def _normalize_v3_result(self, result, sequence_number):
        data = self._result_to_dict(result)
        res = data.get("res", data) if isinstance(data, dict) else {}

        rec_texts = self._as_list(self._first_present(res, "rec_texts", "texts"))
        rec_scores = self._as_list(self._first_present(res, "rec_scores", "scores"))
        rec_boxes = self._as_list(self._first_present(res, "rec_boxes", "rec_polys", "dt_polys"))

        lines = []
        for idx, text in enumerate(rec_texts):
            text = str(text).strip()
            if not text:
                continue
            score = self._safe_float(rec_scores[idx]) if idx < len(rec_scores) else None
            bbox = self._jsonable(rec_boxes[idx]) if idx < len(rec_boxes) else None
            lines.append({"text": text, "score": score, "bbox": bbox, "detector": "paddleocr"})

        lines.sort(key=lambda item: self._bbox_sort_key(item.get("bbox")))
        return self._page_record(sequence_number, lines, page_index=res.get("page_index"))

    def _normalize_v2_lines(self, result):
        rows = self._flatten_v2_rows(result)
        lines = []
        for row in rows:
            bbox, text_score = row[0], row[1]
            text = str(text_score[0]).strip()
            if not text:
                continue
            score = self._safe_float(text_score[1]) if len(text_score) > 1 else None
            lines.append({"text": text, "score": score, "bbox": self._jsonable(bbox), "detector": "paddleocr"})
        lines.sort(key=lambda item: self._bbox_sort_key(item.get("bbox")))
        return lines

    def _flatten_v2_rows(self, value):
        if self._looks_like_v2_row(value):
            return [value]
        rows = []
        if isinstance(value, (list, tuple)):
            for item in value:
                rows.extend(self._flatten_v2_rows(item))
        return rows

    def _looks_like_v2_row(self, value):
        return (
            isinstance(value, (list, tuple))
            and len(value) >= 2
            and isinstance(value[1], (list, tuple))
            and len(value[1]) >= 1
            and isinstance(value[1][0], str)
        )

    def _page_record(self, page_number, lines, page_index=None):
        scores = [line["score"] for line in lines if isinstance(line.get("score"), (int, float))]
        return {
            "page_number": int(page_number),
            "page_index": page_index,
            "text": "\n".join(line["text"] for line in lines if line.get("text")),
            "avg_confidence": round(statistics.mean(scores), 4) if scores else None,
            "lines": lines,
        }

    def _load_pages_from_json(self, json_path):
        if not json_path.exists():
            return []
        try:
            payload = json.loads(json_path.read_text(encoding="utf-8"))
            pages = payload.get("pages", [])
            return pages if isinstance(pages, list) else []
        except json.JSONDecodeError as exc:
            print(f"⚠️ Không đọc được checkpoint OCR, sẽ OCR lại: {json_path} ({exc})")
            return []

    def _write_markdown(self, md_path, book_id, pages):
        md_path.parent.mkdir(parents=True, exist_ok=True)
        parts = [
            f"<!-- OCR_ENGINE: {self._ocr_engine_name()} -->",
            f"<!-- OCR_MODE: {self.ocr_mode} -->",
            f"<!-- OCR_LANG: {self.lang} -->",
            f"<!-- OCR_VERSION: {self.ocr_version} -->",
            f"<!-- BOOK_ID: {book_id} -->",
            f"<!-- CHECKPOINT_UPDATED_AT: {datetime.now().isoformat()} -->",
            "",
        ]
        for page in self._sorted_pages(pages):
            parts.append(f"<!-- OCR_PAGE: {page['page_number']} -->")
            if page.get("avg_confidence") is not None:
                parts.append(f"<!-- OCR_AVG_CONFIDENCE: {page['avg_confidence']} -->")
            parts.append(page.get("text", "").strip())
            parts.append("")
        self._atomic_write_text(md_path, "\n".join(parts))

    def _write_json(self, json_path, book_id, pages, status="running", total_pages=None):
        json_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "book_id": book_id,
            "ocr_engine": self._ocr_engine_name(),
            "ocr_mode": self.ocr_mode,
            "ocr_lang": self.lang,
            "ocr_version": self.ocr_version,
            "vietocr_config": config.VIETOCR_CONFIG if self._use_vietocr() else None,
            "status": status,
            "updated_at": datetime.now().isoformat(),
            "completed_pages": len(pages),
            "total_pages": total_pages,
            "pages": self._sorted_pages(pages),
        }
        self._atomic_write_text(json_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _write_phase_checkpoint(self, book_id, status, current_page, total_pages, json_path, md_path):
        checkpoint_path = Path(config.CHECKPOINT_DIR) / f"{book_id}_phase1.json"
        payload = {
            "phase": "phase1",
            "book_id": book_id,
            "status": status,
            "current_page": current_page,
            "total_pages": total_pages,
            "ocr_engine": self._ocr_engine_name(),
            "ocr_json": str(json_path),
            "markdown": str(md_path),
            "updated_at": datetime.now().isoformat(),
        }
        self._atomic_write_text(checkpoint_path, json.dumps(payload, ensure_ascii=False, indent=2))

    def _atomic_write_text(self, path, text):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        tmp_path.replace(path)

    def _sorted_pages(self, pages):
        return sorted(pages, key=lambda item: item.get("page_number", 0))

    def _stop_requested(self):
        return Path(config.STOP_FILE).exists()

    def _result_to_dict(self, result):
        if isinstance(result, dict):
            return result
        if hasattr(result, "json"):
            data = result.json
            return data() if callable(data) else data
        if hasattr(result, "to_dict"):
            return result.to_dict()
        return {}

    def _first_present(self, mapping, *keys):
        for key in keys:
            if isinstance(mapping, dict) and key in mapping and mapping[key] is not None:
                return mapping[key]
        return []

    def _as_list(self, value):
        value = self._jsonable(value)
        if value is None:
            return []
        return value if isinstance(value, list) else list(value)

    def _jsonable(self, value):
        if hasattr(value, "tolist"):
            return value.tolist()
        if isinstance(value, tuple):
            return [self._jsonable(item) for item in value]
        if isinstance(value, list):
            return [self._jsonable(item) for item in value]
        return value

    def _safe_float(self, value):
        try:
            return round(float(value), 4)
        except (TypeError, ValueError):
            return None

    def _bbox_sort_key(self, bbox):
        rect = self._bbox_rect(bbox)
        if not rect:
            return (0, 0)
        left, top, _right, _bottom = rect
        return (top, left)


if __name__ == "__main__":
    converter = PDFConverter()
    converter.convert(f"{config.CURRENT_BOOK}.pdf")


