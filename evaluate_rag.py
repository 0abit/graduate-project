import os
import sys
from pathlib import Path
import pandas as pd

# Thiết lập đường dẫn thư mục gốc để import src
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các config và hàm sinh từ hệ thống hiện tại
import time
import config
from src.api import embed_query, retrieve_context, generate_answer

# Import cấu hình Ragas với Gemini
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from ragas import evaluate
from ragas.metrics.collections import Faithfulness, AnswerRelevancy, ContextPrecision, ContextRecall
from datasets import Dataset

import logging
logging.basicConfig(level=logging.INFO)

# Thiết lập API Key cho Langchain & Ragas
os.environ["GOOGLE_API_KEY"] = config.GEMINI_KEYS[0]

# Khởi tạo mô hình AI làm Giám khảo (LLM-as-a-judge)
llm = ChatGoogleGenerativeAI(model=config.GEMINI_CHAT_MODEL)
embeddings = GoogleGenerativeAIEmbeddings(model=config.GEMINI_EMBEDDING_MODEL)

# Chạy toàn bộ 50 câu hỏi
NUM_TEST_QUESTIONS = 50 

def run_evaluation():
    # 1. Đọc câu hỏi từ CSV
    csv_path = os.path.join(PROJECT_ROOT, "Test_Dataset_Sinh12.csv")
    if not os.path.exists(csv_path):
        # Kiểm tra ở thư mục brain
        brain_path = r"C:\Users\ADMIN\.gemini\antigravity\brain\b482b551-d325-407e-a639-c1ca62f00b02\Test_Dataset_Sinh12.csv"
        if os.path.exists(brain_path):
            csv_path = brain_path
        else:
            print("Không tìm thấy file Test_Dataset_Sinh12.csv!")
            return

    df = pd.read_csv(csv_path)
    
    # Rút gọn danh sách câu hỏi để test
    questions = df['question'].tolist()[:NUM_TEST_QUESTIONS]
    ground_truths = df['ground_truth'].tolist()[:NUM_TEST_QUESTIONS]
    
    answers = []
    contexts_list = []
    
    # 2. Sinh câu trả lời từ hệ thống RAG hiện tại
    print(f"=== ĐANG CHẠY RAG CHO {len(questions)} CÂU HỎI ===")
    for idx, q in enumerate(questions):
        print(f"[{idx+1}/{len(questions)}] Đang xử lý: {q}")
        try:
            # Fake query embedding
            q_emb = embed_query(q)
            # Retrieve từ Neo4j
            records = retrieve_context(q_emb, top_k=3)
            # Lấy list nội dung text để cho Ragas đánh giá
            ctxs = [rec["noi_dung"] for rec in records]
            # Sinh câu trả lời
            ans = generate_answer(q, records)
        except Exception as e:
            print(f"Lỗi ở câu {idx+1}: {e}")
            ans = "Lỗi sinh câu trả lời"
            ctxs = []
            
        answers.append(ans)
        contexts_list.append(ctxs)
        
        # Nghỉ 60 giây sau mỗi câu hỏi để tuyệt đối an toàn với giới hạn 15 request/phút
        time.sleep(60)
        
    # 3. Tạo Dataset HuggingFace format cho Ragas
    data = {
        "question": questions,
        "answer": answers,
        "contexts": contexts_list,
        "ground_truth": ground_truths
    }
    dataset = Dataset.from_dict(data)
    
    # 4. Chấm điểm bằng Ragas
    print("\n=== BẮT ĐẦU CHẤM ĐIỂM BẰNG RAGAS ===")
    result = evaluate(
        dataset=dataset,
        metrics=[
            ContextPrecision(),
            ContextRecall(),
            Faithfulness(),
            AnswerRelevancy(),
        ],
        llm=llm,
        embeddings=embeddings,
        raise_exceptions=False,
    )
    
    print("\n" + "="*50)
    print("KẾT QUẢ ĐÁNH GIÁ RAGAS:")
    print(result)
    print("="*50)
    
    # 5. Lưu kết quả
    result_df = result.to_pandas()
    out_path = os.path.join(PROJECT_ROOT, "Ragas_Evaluation_Results.csv")
    result_df.to_csv(out_path, index=False)
    print(f"Đã lưu chi tiết bảng điểm vào: {out_path}")

if __name__ == "__main__":
    run_evaluation()
