import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import google.generativeai as genai
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class BatchAPIEnricher:
    def __init__(self):
        self.client = MongoClient(config.MONGO_URI)
        self.collection = self.client[config.DB_NAME][config.COLLECTION_NAME_V2]
        self.keys = config.GEMINI_KEYS
        self.key_group_size = max(1, int(getattr(config, "PHASE3_KEY_GROUP_SIZE", config.BATCH_SIZE)))
        self.group_sleep_time = max(0, int(getattr(config, "PHASE3_GROUP_SLEEP_TIME", config.KEY_SLEEP_TIME)))
        self.all_keys_exhausted_sleep_time = max(
            0,
            int(getattr(config, "PHASE3_ALL_KEYS_EXHAUSTED_SLEEP_TIME", 3600)),
        )
        self.json_retry_count = max(0, int(getattr(config, "PHASE3_JSON_RETRY_COUNT", 2)))
        self.json_retry_sleep_time = max(0, int(getattr(config, "PHASE3_JSON_RETRY_SLEEP_TIME", 5)))
        self.transient_retry_count = max(0, int(getattr(config, "PHASE3_TRANSIENT_RETRY_COUNT", 3)))
        self.transient_retry_sleep_time = max(0, int(getattr(config, "PHASE3_TRANSIENT_RETRY_SLEEP_TIME", 30)))
        self.skip_chunk_on_json_error = bool(getattr(config, "PHASE3_SKIP_CHUNK_ON_JSON_ERROR", True))

    def start_enrichment(self):
        print(
            "💎 [PHASE 3] Bắt đầu xử lý theo nhóm key "
            f"(group size: {self.key_group_size}, nghỉ giữa các lượt: {self.group_sleep_time}s, "
            f"nghỉ khi hết toàn bộ key: {self.all_keys_exhausted_sleep_time}s)"
        )
        if not self.keys:
            print("⚠️ Chưa có GEMINI_KEYS trong .env, bỏ qua Phase 3.")
            return
        print(f"🔑 Đã load {len(self.keys)} Gemini key, chia thành các nhóm {self.key_group_size} key.")

        query = {"metadata.api_refined": False, "metadata.active": True}
        unprocessed = list(self.collection.find(query).sort("metadata.global_chunk_index", 1))
        if not unprocessed:
            print("🎉 Toàn bộ active chunks đã được API xử lý xong.")
            return

        total = len(unprocessed)
        chunk_index = 0
        group_start = 0
        exhausted_key_indexes = set()
        invalid_key_indexes = set()
        skipped_chunks = 0

        try:
            while chunk_index < total:
                if group_start >= len(self.keys):
                    if len(invalid_key_indexes) >= len(self.keys):
                        chunk_id = unprocessed[chunk_index].get("_id") if chunk_index < total else None
                        message = "Tất cả Gemini API key đều không hợp lệ hoặc không có quyền gọi model."
                        self._write_phase_checkpoint(
                            "all_api_keys_invalid",
                            chunk_index,
                            total,
                            chunk_id,
                            "stopped_no_valid_api_key",
                            message,
                        )
                        print(f"🔴 {message} Hãy kiểm tra lại GEMINI_KEYS trong .env rồi chạy lại phase3.py.")
                        return

                    chunk_id = unprocessed[chunk_index].get("_id") if chunk_index < total else None
                    message = "Tất cả Gemini key còn hợp lệ hiện có đều đã hết quota hoặc bị rate limit."
                    self._write_phase_checkpoint(
                        "all_keys_exhausted_sleeping",
                        chunk_index,
                        total,
                        chunk_id,
                        "sleeping_until_quota_reset",
                        message,
                    )
                    print(
                        f"🟡 {message} Nghỉ {self.all_keys_exhausted_sleep_time}s, "
                        "sau đó dò lại từ key 1."
                    )
                    self._sleep_with_stop(self.all_keys_exhausted_sleep_time)
                    exhausted_key_indexes.clear()
                    group_start = 0
                    self._write_phase_checkpoint(
                        "retrying_after_quota_sleep",
                        chunk_index,
                        total,
                        chunk_id,
                        "retrying_from_first_key_group",
                    )
                    continue

                group_end = min(group_start + self.key_group_size, len(self.keys))
                active_key_indexes = [
                    key_index for key_index in range(group_start, group_end)
                    if key_index not in exhausted_key_indexes and key_index not in invalid_key_indexes
                ]

                if not active_key_indexes:
                    previous_group = self._group_label(group_start)
                    group_start += self.key_group_size
                    if group_start < len(self.keys):
                        print(f"🔁 Nhóm key {previous_group} không còn key khả dụng; chuyển sang nhóm key {self._group_label(group_start)}.")
                    continue

                print(
                    f"--- Lượt key {self._group_label(group_start)} "
                    f"| còn {total - chunk_index} chunk ---"
                )

                made_progress = False
                for absolute_key_index in active_key_indexes:
                    if chunk_index >= total:
                        break
                    if self._stop_requested():
                        raise KeyboardInterrupt

                    chunk = unprocessed[chunk_index]
                    current_key = self.keys[absolute_key_index]
                    key_number = absolute_key_index + 1
                    metadata = chunk.get("metadata", {})
                    chunk_id = chunk.get("_id")

                    print(
                        f"  👉 Chunk {chunk_index + 1}/{total} | "
                        f"Bài: {metadata.get('lesson_title', 'Không rõ')} | "
                        f"Mục: {metadata.get('section_title', 'Nội dung')} | Key: {key_number}"
                    )

                    try:
                        prompt = self._build_prompt(chunk)
                        genai.configure(api_key=current_key)
                        model = genai.GenerativeModel(
                            config.GEMINI_MODEL,
                            generation_config={"response_mime_type": "application/json"},
                        )

                        response_json = self._generate_json_with_retries(model, prompt, key_number)

                        self.collection.update_one(
                            {"_id": chunk_id},
                            {"$set": {
                                "extracted_entities": response_json,
                                "metadata.api_refined": True,
                                "metadata.refined_at": datetime.now().isoformat(),
                                "metadata.refinement_model": config.GEMINI_MODEL,
                                "metadata.phase3_status": "done",
                            }},
                        )
                        chunk_index += 1
                        made_progress = True
                        self._write_phase_checkpoint("running", chunk_index, total, chunk_id, "done")
                        print("     ✅ Trích xuất thành công!")

                    except Exception as exc:
                        error_summary = self._api_error_summary(exc)
                        if self._is_quota_error(exc):
                            exhausted_key_indexes.add(absolute_key_index)
                            self._write_phase_checkpoint(
                                "key_quota_exhausted",
                                chunk_index,
                                total,
                                chunk_id,
                                f"key_{key_number}_quota_exhausted",
                                error_summary,
                            )
                            print(f"     🟡 Key {key_number} hết quota/rate limit; thử key kế tiếp trong nhóm.")
                            print(f"        Chi tiết lỗi: {error_summary}")
                            continue

                        if self._is_api_key_error(exc):
                            invalid_key_indexes.add(absolute_key_index)
                            self._write_phase_checkpoint(
                                "api_key_invalid",
                                chunk_index,
                                total,
                                chunk_id,
                                f"key_{key_number}_invalid",
                                error_summary,
                            )
                            print(f"     🔴 Key {key_number} lỗi xác thực/không có quyền; bỏ qua key này.")
                            print(f"        Chi tiết lỗi: {error_summary}")
                            continue

                        if self._is_transient_error(exc):
                            self._write_phase_checkpoint(
                                "transient_api_error",
                                chunk_index,
                                total,
                                chunk_id,
                                f"key_{key_number}_transient_error",
                                error_summary,
                            )
                            print(f"     🟠 Key {key_number} gặp lỗi tạm thời; thử key kế tiếp.")
                            print(f"        Chi tiết lỗi: {error_summary}")
                            continue

                        if self.skip_chunk_on_json_error and self._is_json_response_error(exc):
                            skipped_chunks += 1
                            self.collection.update_one(
                                {"_id": chunk_id},
                                {"$set": {
                                    "metadata.phase3_status": "json_error_skipped",
                                    "metadata.phase3_error": error_summary,
                                    "metadata.phase3_error_type": type(exc).__name__,
                                    "metadata.phase3_error_at": datetime.now().isoformat(),
                                    "metadata.api_refined": False,
                                }},
                            )
                            chunk_index += 1
                            made_progress = True
                            self._write_phase_checkpoint(
                                "json_error_skipped",
                                chunk_index,
                                total,
                                chunk_id,
                                "skipped_bad_json_response",
                                error_summary,
                            )
                            print("     🟠 Gemini trả JSON lỗi sau nhiều lần retry; đã bỏ qua chunk này để pipeline chạy tiếp.")
                            print(f"        Chi tiết lỗi: {error_summary}")
                            continue

                        self.collection.update_one(
                            {"_id": chunk_id},
                            {"$set": {
                                "metadata.phase3_status": "error",
                                "metadata.phase3_error": error_summary,
                                "metadata.phase3_error_type": type(exc).__name__,
                                "metadata.phase3_error_at": datetime.now().isoformat(),
                            }},
                        )
                        self._write_phase_checkpoint("error", chunk_index + 1, total, chunk_id, "error", error_summary)
                        print("     ❌ Lỗi API/chunk chưa được tự xử lý, dừng phase để bảo vệ dữ liệu/key.")
                        print(f"        Loại lỗi: {type(exc).__name__}")
                        print(f"        Chi tiết lỗi: {error_summary}")
                        raise

                active_key_indexes = [
                    key_index for key_index in range(group_start, group_end)
                    if key_index not in exhausted_key_indexes and key_index not in invalid_key_indexes
                ]
                if not active_key_indexes:
                    previous_group = self._group_label(group_start)
                    group_start += self.key_group_size
                    if group_start < len(self.keys):
                        print(f"🔁 Nhóm key {previous_group} không còn key khả dụng; chuyển sang nhóm key {self._group_label(group_start)}.")
                    continue

                if chunk_index < total and made_progress:
                    print(f"⏳ Nghỉ {self.group_sleep_time}s rồi dùng lại nhóm key {self._group_label(group_start)}...")
                    self._sleep_with_stop(self.group_sleep_time)
                elif chunk_index < total and not made_progress:
                    wait_seconds = self.transient_retry_sleep_time or self.group_sleep_time or 60
                    print(f"⏳ Chưa xử lý được chunk hiện tại trong lượt này; nghỉ {wait_seconds}s rồi thử tiếp...")
                    self._sleep_with_stop(wait_seconds)

        except KeyboardInterrupt:
            print("🛑 Phase 3 đã dừng an toàn. Chạy lại phase3.py để tiếp tục các chunk chưa api_refined.")
            raise

        final_status = "complete_with_skipped_chunks" if skipped_chunks else "complete"
        self._write_phase_checkpoint(final_status, total, total, None, final_status)
        if skipped_chunks:
            print(f"⚠️ Hoàn tất phần tự động, nhưng có {skipped_chunks} chunk bị bỏ qua vì JSON lỗi. Lọc metadata.phase3_status='json_error_skipped' để xử lý lại.")

    def _build_prompt(self, chunk):
        metadata = chunk.get("metadata", {})
        return f"""
Bạn là chuyên gia giáo dục môn Sinh học phổ thông Việt Nam.
Đọc đoạn tài liệu OCR dưới đây và trích xuất thông tin cốt lõi.
Văn bản có thể còn lỗi OCR nhẹ về dấu tiếng Việt, khoảng trắng hoặc chữ hoa/thường; hãy chuẩn hóa nhẹ khi trích xuất thực thể, nhưng không được bịa nội dung không có trong đoạn.
NẾU trong văn bản có chứa thẻ hình ảnh dạng `![Hình ảnh minh hoạ](...)`, BẮT BUỘC phải giữ nguyên và truyền thẻ đó vào các trường phù hợp hoặc để nguyên trong nội dung, tuyệt đối không được xoá mất đường dẫn ảnh.

Bối cảnh metadata:
- Sách: {metadata.get('book_id', '')}
- Bài học: {metadata.get('lesson_title', '')}
- Mục: {metadata.get('section_title', '')}
- Yêu cầu cần đạt: {json.dumps(metadata.get('learning_objectives', []), ensure_ascii=False)}
- Trang nguồn: {metadata.get('source_pages', [])}

Tuyệt đối trả về JSON đúng cấu trúc dưới đây. Nếu không có thông tin cho một mục, trả về mảng rỗng [].

{{
  "khai_niem_dinh_nghia": ["Tên các khái niệm mới. VD: Gene, mã di truyền..."],
  "cau_truc_chuc_nang": ["Các chi tiết về cấu tạo hoặc chức năng"],
  "co_che_qua_trinh": ["Tên các cơ chế/quá trình sinh học. VD: tái bản DNA, phiên mã..."],
  "thi_nghiem_chung_minh": ["Tên thí nghiệm hoặc người thực hiện"],
  "ung_dung_thanh_tuu": ["Các ứng dụng/thành tựu thực tiễn"],
  "y_nghia_sinh_hoc": ["Ý nghĩa sinh học của quá trình/hiện tượng"],
  "tom_tat_cot_loi": "Tóm tắt nội dung đoạn trong 1-2 câu ngắn gọn bằng tiếng Việt chuẩn."
}}

Văn bản OCR:
{chunk.get('markdown_content', '')}
"""

    def _generate_json_with_retries(self, model, prompt, key_number):
        attempts = self.json_retry_count + 1
        last_error = None
        last_text = ""

        for attempt in range(1, attempts + 1):
            current_prompt = prompt if attempt == 1 else self._json_retry_prompt(prompt, last_text, last_error)

            try:
                response = self._generate_content_with_transient_retries(model, current_prompt, key_number)
                last_text = self._response_text(response)
                return self._normalize_response_json(self._parse_json_response(last_text))
            except (json.JSONDecodeError, ValueError, TypeError) as exc:
                last_error = exc
                print(
                    f"     ⚠️ Key {key_number} trả JSON không hợp lệ "
                    f"(lần {attempt}/{attempts}): {self._short_error(exc)}"
                )
                preview = self._text_preview(last_text)
                if preview:
                    print(f"        Preview response: {preview}")
                if attempt < attempts and self.json_retry_sleep_time:
                    self._sleep_with_stop(self.json_retry_sleep_time)

        raise ValueError(
            "Gemini trả JSON không hợp lệ sau "
            f"{attempts} lần. Lỗi cuối: {self._short_error(last_error)}. "
            f"Response cuối: {self._text_preview(last_text, limit=500)}"
        )

    def _generate_content_with_transient_retries(self, model, prompt, key_number):
        attempts = self.transient_retry_count + 1
        last_error = None
        for attempt in range(1, attempts + 1):
            try:
                return model.generate_content(prompt)
            except Exception as exc:
                if self._is_quota_error(exc) or self._is_api_key_error(exc) or not self._is_transient_error(exc):
                    raise
                last_error = exc
                print(
                    f"     🟠 Key {key_number} lỗi tạm thời "
                    f"(lần {attempt}/{attempts}): {self._api_error_summary(exc)}"
                )
                if attempt < attempts and self.transient_retry_sleep_time:
                    self._sleep_with_stop(self.transient_retry_sleep_time)
        raise last_error

    def _json_retry_prompt(self, original_prompt, bad_response, error):
        return f"""
Lần trả lời trước không parse được JSON.
Lỗi parser: {self._short_error(error)}

Hãy trả lời lại bằng JSON thuần, không markdown, không ```json, không giải thích ngoài JSON.
JSON bắt buộc có đúng các key:
- khai_niem_dinh_nghia
- cau_truc_chuc_nang
- co_che_qua_trinh
- thi_nghiem_chung_minh
- ung_dung_thanh_tuu
- y_nghia_sinh_hoc
- tom_tat_cot_loi

Nếu một mục không có thông tin, dùng [] hoặc chuỗi rỗng cho tom_tat_cot_loi.

Response lỗi trước đó:
{self._text_preview(bad_response, limit=1200)}

Yêu cầu gốc:
{original_prompt}
"""

    def _response_text(self, response):
        try:
            text = response.text
        except Exception as exc:
            raise ValueError(f"Gemini response không có text hợp lệ: {self._short_error(exc)}") from exc

        if text is None or not str(text).strip():
            raise ValueError("Gemini response rỗng.")
        return str(text)

    def _parse_json_response(self, text):
        if text is None:
            raise ValueError("Gemini response rỗng.")
        text = str(text).strip()
        if not text:
            raise ValueError("Gemini response rỗng.")

        text = self._strip_markdown_fence(text)
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            candidate = self._extract_balanced_json_object(text)
            if not candidate:
                raise

        try:
            return json.loads(candidate)
        except json.JSONDecodeError:
            repaired = self._repair_common_json_issues(candidate)
            return json.loads(repaired)

    def _strip_markdown_fence(self, text):
        text = text.strip()
        fence_match = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.DOTALL | re.IGNORECASE)
        return fence_match.group(1).strip() if fence_match else text

    def _extract_balanced_json_object(self, text):
        start = text.find("{")
        if start < 0:
            return None

        depth = 0
        in_string = False
        escape = False
        for index in range(start, len(text)):
            char = text[index]

            if escape:
                escape = False
                continue
            if char == "\\" and in_string:
                escape = True
                continue
            if char == '"':
                in_string = not in_string
                continue
            if in_string:
                continue
            if char == "{":
                depth += 1
            elif char == "}":
                depth -= 1
                if depth == 0:
                    return text[start:index + 1]
        return None

    def _repair_common_json_issues(self, text):
        text = re.sub(r",\s*([}\]])", r"\1", text)
        text = text.replace("\ufeff", "").strip()
        return text

    def _normalize_response_json(self, value):
        if not isinstance(value, dict):
            raise ValueError("JSON Gemini trả về không phải object.")

        list_keys = (
            "khai_niem_dinh_nghia",
            "cau_truc_chuc_nang",
            "co_che_qua_trinh",
            "thi_nghiem_chung_minh",
            "ung_dung_thanh_tuu",
            "y_nghia_sinh_hoc",
        )
        normalized = dict(value)
        for key in list_keys:
            item = normalized.get(key, [])
            if item is None:
                item = []
            elif isinstance(item, str):
                item = [item] if item.strip() else []
            elif not isinstance(item, list):
                item = [str(item)]
            normalized[key] = item

        summary = normalized.get("tom_tat_cot_loi", "")
        if summary is None:
            summary = ""
        elif not isinstance(summary, str):
            summary = str(summary)
        normalized["tom_tat_cot_loi"] = summary
        return normalized

    def _is_quota_error(self, exc):
        text = self._exception_text(exc)
        quota_markers = (
            "429",
            "quota",
            "rate limit",
            "rate_limit",
            "ratelimit",
            "rate exceeded",
            "rateexceeded",
            "resource exhausted",
            "resource_exhausted",
            "resourceexhausted",
            "too many requests",
            "too_many_requests",
            "per minute",
            "per day",
            "free tier",
            "try again later",
        )
        return any(marker in text for marker in quota_markers)

    def _is_api_key_error(self, exc):
        text = self._exception_text(exc)
        api_key_markers = (
            "api key not valid",
            "api_key_invalid",
            "invalid api key",
            "invalid_api_key",
            "api key expired",
            "api key revoked",
            "authentication",
            "unauthenticated",
            "unauthorized",
            "permission denied",
            "permission_denied",
            "forbidden",
            "403",
            "401",
        )
        return any(marker in text for marker in api_key_markers)

    def _is_transient_error(self, exc):
        text = self._exception_text(exc)
        transient_markers = (
            "timeout",
            "deadline exceeded",
            "deadline_exceeded",
            "temporarily unavailable",
            "unavailable",
            "service unavailable",
            "internal server error",
            "connection reset",
            "connection aborted",
            "connection refused",
            "remote disconnected",
            "dns",
            "ssl",
            "socket",
            "503",
            "504",
            "500",
            "502",
        )
        return any(marker in text for marker in transient_markers)

    def _is_json_response_error(self, exc):
        text = self._exception_text(exc)
        json_markers = (
            "json",
            "parse",
            "parser",
            "decode",
            "không hợp lệ",
            "response rỗng",
            "không có text hợp lệ",
            "không phải object",
        )
        return isinstance(exc, (json.JSONDecodeError, ValueError, TypeError)) and any(
            marker in text for marker in json_markers
        )

    def _exception_text(self, exc):
        return self._sanitize_error_text(self._raw_exception_text(exc)).lower()

    def _api_error_summary(self, exc):
        meta = self._exception_meta(exc)
        text = self._sanitize_error_text(self._raw_exception_text(exc))
        text = re.sub(r"\s+", " ", text).strip()
        if len(text) > 800:
            text = text[:800].rstrip() + "..."

        parts = [f"type={type(exc).__name__}"]
        if meta["code"]:
            parts.append(f"code={meta['code']}")
        if meta["status_code"] and meta["status_code"] != meta["code"]:
            parts.append(f"status_code={meta['status_code']}")
        if meta["status"]:
            parts.append(f"status={meta['status']}")
        if meta["reason"]:
            parts.append(f"reason={meta['reason']}")
        parts.append(f"message={text}")
        return " | ".join(parts)

    def _raw_exception_text(self, exc):
        meta = self._exception_meta(exc)
        status_parts = [
            str(meta["code"] or ""),
            str(meta["status_code"] or ""),
            str(meta["status"] or ""),
            str(meta["reason"] or ""),
        ]
        return " ".join([f"{type(exc).__name__}: {exc}"] + status_parts)

    def _exception_meta(self, exc):
        code = getattr(exc, "code", None)
        status_code = getattr(exc, "status_code", None)
        if callable(code):
            try:
                code = code()
            except TypeError:
                code = None

        return {
            "code": code,
            "status_code": status_code,
            "status": getattr(exc, "status", "") or "",
            "reason": getattr(exc, "reason", "") or "",
        }

    def _sanitize_error_text(self, text):
        text = str(text)
        text = re.sub(r"AIza[0-9A-Za-z_\-]{20,}", "[REDACTED_API_KEY]", text)
        for key in self.keys:
            if key:
                text = text.replace(key, "[REDACTED_API_KEY]")
        return text

    def _short_error(self, exc):
        if exc is None:
            return ""
        text = self._sanitize_error_text(f"{type(exc).__name__}: {exc}")
        text = re.sub(r"\s+", " ", text).strip()
        return text[:300] + "..." if len(text) > 300 else text

    def _text_preview(self, text, limit=300):
        if text is None:
            return ""
        preview = self._sanitize_error_text(text)
        preview = re.sub(r"\s+", " ", preview).strip()
        return preview[:limit].rstrip() + "..." if len(preview) > limit else preview

    def _group_label(self, group_start):
        group_end = min(group_start + self.key_group_size, len(self.keys))
        return f"{group_start + 1}-{group_end}"

    def _write_phase_checkpoint(self, status, processed_chunks, total_chunks, chunk_id, chunk_status, error=None):
        checkpoint_path = Path(config.CHECKPOINT_DIR) / "phase3_enrichment.json"
        payload = {
            "phase": "phase3",
            "status": status,
            "processed_chunks": processed_chunks,
            "total_chunks": total_chunks,
            "chunk_id": str(chunk_id) if chunk_id is not None else None,
            "chunk_status": chunk_status,
            "error": error,
            "updated_at": datetime.now().isoformat(),
        }
        text = json.dumps(payload, ensure_ascii=False, indent=2)
        try:
            self._atomic_write_text(checkpoint_path, text)
        except OSError as exc:
            fallback_path = checkpoint_path.with_name(
                f"{checkpoint_path.stem}.{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )
            try:
                fallback_path.write_text(text, encoding="utf-8")
                print(
                    "⚠️ Không ghi đè được checkpoint chính "
                    f"({self._short_error(exc)}); đã ghi fallback: {fallback_path}"
                )
            except OSError as fallback_exc:
                print(
                    "⚠️ Không ghi được checkpoint Phase 3, nhưng vẫn tiếp tục chạy. "
                    f"Lỗi chính: {self._short_error(exc)} | Lỗi fallback: {self._short_error(fallback_exc)}"
                )

    def _sleep_with_stop(self, seconds):
        for _ in range(seconds):
            if self._stop_requested():
                raise KeyboardInterrupt
            time.sleep(1)

    def _stop_requested(self):
        return Path(config.STOP_FILE).exists()

    def _atomic_write_text(self, path, text):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_name(f"{path.name}.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        last_error = None
        for _ in range(5):
            try:
                tmp_path.replace(path)
                return
            except PermissionError as exc:
                last_error = exc
                time.sleep(0.25)
        if last_error:
            raise last_error


if __name__ == "__main__":
    enricher = BatchAPIEnricher()
    enricher.start_enrichment()

