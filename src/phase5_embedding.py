import os
import sys
import time
from pathlib import Path

import google.generativeai as genai
from neo4j import GraphDatabase

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

class EmbeddingGenerator:
    def __init__(self):
        self.driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))
        
        self.keys = config.GEMINI_KEYS
        if not self.keys:
            raise ValueError("Không tìm thấy GEMINI_KEYS trong config.")
        genai.configure(api_key=self.keys[0])

    def setup_vector_index(self):
        print("🔧 Cài đặt Vector Index trong Neo4j...")
        query = """
        CREATE VECTOR INDEX PhanDoanV2_embedding IF NOT EXISTS
        FOR (n:PhanDoanV2) ON (n.embedding)
        OPTIONS {indexConfig: {
            `vector.dimensions`: 3072,
            `vector.similarity_function`: 'cosine'
        }}
        """
        with self.driver.session() as session:
            session.run(query)
        print("✅ Đã thiết lập Vector Index.")

    def run(self):
        self.setup_vector_index()
        
        print("🚀 Bắt đầu tạo embeddings cho các Phân Đoạn...")
        
        # Lấy các chunk chưa có embedding
        query_unembedded = """
        MATCH (p:PhanDoanV2)
        WHERE p.embedding IS NULL AND p.noi_dung IS NOT NULL AND p.noi_dung <> ""
        RETURN p.ma_phan_doan AS node_id, p.noi_dung AS noi_dung
        """
        
        with self.driver.session() as session:
            result = session.run(query_unembedded)
            nodes = [record.data() for record in result]
            
        if not nodes:
            print("🎉 Tất cả các Phân Đoạn đều đã có embedding.")
            return

        total = len(nodes)
        print(f"📌 Tìm thấy {total} Phân Đoạn cần tạo embedding.")
        
        # Xử lý theo lô để tối ưu và tránh rate limit
        batch_size = config.PHASE5_BATCH_SIZE
        
        for i in range(0, total, batch_size):
            batch = nodes[i:i + batch_size]
            texts = [node["noi_dung"] for node in batch]
            
            try:
                # Gọi Gemini API để lấy embeddings
                response = genai.embed_content(
                    model=config.GEMINI_EMBEDDING_MODEL,
                    content=texts,
                    task_type="retrieval_document"
                )
                
                embeddings = response['embedding']
                
                # Cập nhật vào Neo4j
                update_query = """
                UNWIND $batch_data AS data
                MATCH (p:PhanDoanV2 {ma_phan_doan: data.node_id})
                SET p.embedding = data.embedding
                """
                
                batch_data = [
                    {"node_id": node["node_id"], "embedding": emb}
                    for node, emb in zip(batch, embeddings)
                ]
                
                with self.driver.session() as session:
                    session.run(update_query, batch_data=batch_data)
                
                print(f"✅ Đã xử lý {min(i + batch_size, total)}/{total} chunks.")
                
                # Nghỉ ngắn để tránh rate limit
                time.sleep(config.PHASE5_SLEEP_TIME)
                
            except Exception as e:
                print(f"❌ Lỗi khi xử lý batch từ {i}: {e}")
                time.sleep(config.PHASE5_ERROR_SLEEP_TIME) # Đợi lâu hơn nếu có lỗi
                
        print("🎉 Hoàn tất quá trình tạo embedding!")
        self.driver.close()

if __name__ == "__main__":
    generator = EmbeddingGenerator()
    generator.run()
