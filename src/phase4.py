import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

from neo4j import GraphDatabase
from pymongo import MongoClient

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config


class Neo4jIngestor:
    ENTITY_LIST_KEYS = (
        "khai_niem_dinh_nghia",
        "cau_truc_chuc_nang",
        "co_che_qua_trinh",
        "thi_nghiem_chung_minh",
        "ung_dung_thanh_tuu",
        "y_nghia_sinh_hoc",
    )

    def __init__(self):
        self.driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

        mongo_client = MongoClient(config.MONGO_URI)
        self.collection = mongo_client[config.DB_NAME][config.COLLECTION_NAME_V2]

    def _ingest_chunk(self, chunk_data):
        query = """
        MERGE (s:Sach {ma_sach: $ma_sach})
        MERGE (b:BaiHocV2 {ma_bai_hoc: $ma_bai_hoc_unique})
        SET b.ten_hien_thi = $ten_bai_hoc,
            b.lesson_id = $lesson_id,
            b.grade = $grade,
            b.source_level = $source_level,
            b.learning_objectives = $learning_objectives,
            b.page_start = $lesson_page_start,
            b.page_end = $lesson_page_end
        MERGE (s)-[:BAO_GOM]->(b)

        MERGE (p:PhanDoanV2 {ma_phan_doan: $ma_phan_doan})
        SET p.noi_dung = $noi_dung,
            p.tieu_muc = $tieu_muc,
            p.tom_tat = $tom_tat,
            p.source_pages = $source_pages,
            p.chunk_index = $chunk_index,
            p.parser = $parser
        MERGE (b)-[:CO_PHAN_DOAN]->(p)

        FOREACH (item IN $kn_list | MERGE (kn:KhaiNiemV2 {ten: item}) MERGE (p)-[:DINH_NGHIA]->(kn))
        FOREACH (item IN $ct_list | MERGE (ct:CauTrucChucNang {ten: item}) MERGE (p)-[:MO_TA_CAU_TRUC_CHUC_NANG]->(ct))
        FOREACH (item IN $cc_list | MERGE (cc:CoChe {ten: item}) MERGE (p)-[:GIAI_THICH_CO_CHE]->(cc))
        FOREACH (item IN $tn_list | MERGE (tn:ThiNghiem {ten: item}) MERGE (p)-[:CHUNG_MINH_BANG_THI_NGHIEM]->(tn))
        FOREACH (item IN $ud_list | MERGE (ud:UngDung {ten: item}) MERGE (p)-[:CO_UNG_DUNG_THUC_TIEN]->(ud))
        FOREACH (item IN $yn_list | MERGE (yn:YNghia {ten: item}) MERGE (p)-[:MANG_Y_NGHIA_SINH_HOC]->(yn))
        """

        metadata = chunk_data.get("metadata", {})
        ext = self._normalize_extracted_entities(chunk_data.get("extracted_entities", {}))
        lesson_key = metadata.get("lesson_id") or metadata.get("lesson_title", "unknown")

        parameters = {
            "ma_sach": metadata.get("book_id"),
            "ma_bai_hoc_unique": f"{metadata.get('book_id')}_{lesson_key}",
            "lesson_id": metadata.get("lesson_id"),
            "grade": metadata.get("grade"),
            "source_level": metadata.get("source_level", "textbook"),
            "learning_objectives": metadata.get("learning_objectives", []),
            "ten_bai_hoc": metadata.get("lesson_title", ""),
            "lesson_page_start": metadata.get("lesson_page_start"),
            "lesson_page_end": metadata.get("lesson_page_end"),
            "tieu_muc": metadata.get("section_title", ""),
            "source_pages": metadata.get("source_pages", []),
            "chunk_index": metadata.get("chunk_index"),
            "parser": metadata.get("parser", "unknown"),
            "ma_phan_doan": str(chunk_data["_id"]),
            "noi_dung": chunk_data.get("markdown_content", ""),
            "tom_tat": ext["tom_tat_cot_loi"],
            "kn_list": ext["khai_niem_dinh_nghia"],
            "ct_list": ext["cau_truc_chuc_nang"],
            "cc_list": ext["co_che_qua_trinh"],
            "tn_list": ext["thi_nghiem_chung_minh"],
            "ud_list": ext["ung_dung_thanh_tuu"],
            "yn_list": ext["y_nghia_sinh_hoc"],
        }

        with self.driver.session() as session:
            session.run(query, parameters)

    def _normalize_extracted_entities(self, raw):
        normalized = {key: [] for key in self.ENTITY_LIST_KEYS}
        normalized["tom_tat_cot_loi"] = ""

        if isinstance(raw, dict):
            self._merge_entity_dict(normalized, raw)
        elif isinstance(raw, list):
            summaries = []
            for item in raw:
                if isinstance(item, dict):
                    self._merge_entity_dict(normalized, item)
                    summary_text = self._stringify_entity_value(item.get("tom_tat_cot_loi", ""))
                    if summary_text:
                        summaries.append(summary_text)
                elif item is not None:
                    summaries.append(self._stringify_entity_value(item))
            if summaries:
                normalized["tom_tat_cot_loi"] = " ".join(self._dedupe_strings(summaries))
        elif raw is not None:
            normalized["tom_tat_cot_loi"] = self._stringify_entity_value(raw)

        for key in self.ENTITY_LIST_KEYS:
            normalized[key] = self._dedupe_strings(normalized[key])
        normalized["tom_tat_cot_loi"] = str(normalized.get("tom_tat_cot_loi") or "").strip()
        return normalized

    def _merge_entity_dict(self, target, source):
        for key in self.ENTITY_LIST_KEYS:
            target[key].extend(self._as_clean_string_list(source.get(key, [])))

        summary = source.get("tom_tat_cot_loi", "")
        summary_text = self._stringify_entity_value(summary)
        if summary_text:
            existing = target.get("tom_tat_cot_loi", "")
            target["tom_tat_cot_loi"] = f"{existing} {summary_text}".strip() if existing else summary_text

    def _as_clean_string_list(self, value):
        if value is None:
            return []
        if isinstance(value, list):
            items = value
        else:
            items = [value]

        cleaned = []
        for item in items:
            text = self._stringify_entity_value(item)
            if text:
                cleaned.append(text)
        return cleaned

    def _stringify_entity_value(self, value):
        if value is None:
            return ""
        if isinstance(value, str):
            return value.strip()
        if isinstance(value, (int, float, bool)):
            return str(value).strip()
        if isinstance(value, dict):
            for key in ("ten", "name", "title", "khai_niem", "noi_dung", "text"):
                if value.get(key):
                    return str(value[key]).strip()
            return json.dumps(value, ensure_ascii=False, sort_keys=True).strip()
        if isinstance(value, list):
            return "; ".join(filter(None, (self._stringify_entity_value(item) for item in value))).strip()
        return str(value).strip()

    def _dedupe_strings(self, values):
        seen = set()
        deduped = []
        for value in values:
            text = self._stringify_entity_value(value)
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            deduped.append(text)
        return deduped

    def run_sync(self):
        print("🚀 [PHASE 4] Bắt đầu đồng bộ đồ thị Neo4j...")
        query = {
            "metadata.api_refined": True,
            "metadata.active": True,
            "metadata.neo4j_synced": {"$ne": True},
        }
        refined_chunks = list(self.collection.find(query).sort("metadata.global_chunk_index", 1))

        if not refined_chunks:
            print("🎉 Không còn active chunk nào cần sync Neo4j.")
            self.driver.close()
            return

        total = len(refined_chunks)
        try:
            for i, chunk in enumerate(refined_chunks, start=1):
                if self._stop_requested():
                    raise KeyboardInterrupt

                self._ingest_chunk(chunk)
                self.collection.update_one(
                    {"_id": chunk["_id"]},
                    {"$set": {
                        "metadata.neo4j_synced": True,
                        "metadata.neo4j_synced_at": datetime.now().isoformat(),
                        "metadata.phase4_status": "done",
                    }},
                )
                self._write_phase_checkpoint("running", i, total, chunk["_id"], "done")
                metadata = chunk.get("metadata", {})
                print(f"✅ Đã dệt đồ thị [{i}/{total}]: {metadata.get('lesson_title', 'Không rõ')}")
        except KeyboardInterrupt:
            print("🛑 Phase 4 đã dừng an toàn. Chạy lại phase4.py để sync tiếp các chunk chưa neo4j_synced.")
            raise
        except Exception as exc:
            chunk_id = chunk.get("_id") if "chunk" in locals() else None
            if chunk_id is not None:
                self.collection.update_one(
                    {"_id": chunk_id},
                    {"$set": {
                        "metadata.phase4_status": "error",
                        "metadata.phase4_error": str(exc),
                        "metadata.phase4_error_at": datetime.now().isoformat(),
                    }},
                )
            self._write_phase_checkpoint("error", i if "i" in locals() else 0, total, chunk_id, "error", str(exc))
            raise
        finally:
            self.driver.close()

        self._write_phase_checkpoint("complete", total, total, None, "complete")
        print("🎉 Hoàn tất xây dựng Neo4j!")

    def _write_phase_checkpoint(self, status, processed_chunks, total_chunks, chunk_id, chunk_status, error=None):
        checkpoint_path = Path(config.CHECKPOINT_DIR) / "phase4_neo4j.json"
        payload = {
            "phase": "phase4",
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
                print(f"⚠️ Không ghi đè được checkpoint chính ({exc}); đã ghi fallback: {fallback_path}")
            except OSError as fallback_exc:
                print(
                    "⚠️ Không ghi được checkpoint Phase 4, nhưng vẫn tiếp tục chạy. "
                    f"Lỗi chính: {exc} | Lỗi fallback: {fallback_exc}"
                )

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
    ingestor = Neo4jIngestor()
    ingestor.run_sync()
