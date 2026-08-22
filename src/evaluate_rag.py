import os
import sys
import time
from pathlib import Path
import pandas as pd

# Thiết lập đường dẫn thư mục gốc để import src
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import các config và hàm sinh từ hệ thống hiện tại
from src import config
from src.api import embed_query, retrieve_context, generate_answer

# Import cấu hình Ragas với Gemini
import google.generativeai as genai
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings, chat_models

# --- MONKEY PATCH FIX CHO LANGCHAIN-GOOGLE-GENAI ---
# 1. Ragas truyền 'temperature' vào kwargs nhưng thư viện google-genai bản cũ không hiểu, gây ra lỗi.
original_generate = ChatGoogleGenerativeAI._generate
def patched_generate(self, messages, stop=None, run_manager=None, **kwargs):
    kwargs.pop("temperature", None)
    return original_generate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
ChatGoogleGenerativeAI._generate = patched_generate

original_agenerate = ChatGoogleGenerativeAI._agenerate
async def patched_agenerate(self, messages, stop=None, run_manager=None, **kwargs):
    kwargs.pop("temperature", None)
    return await original_agenerate(self, messages, stop=stop, run_manager=run_manager, **kwargs)
ChatGoogleGenerativeAI._agenerate = patched_agenerate

# 2. Vô hiệu hóa tính năng tự động Retry khi bị lỗi 429 của Langchain
# Langchain mặc định sẽ tự retry rất lâu nếu bị 429 Quota Exceeded thay vì văng lỗi ra ngoài.
def dummy_retry_decorator():
    def decorator(func):
        return func
    return decorator
chat_models._create_retry_decorator = dummy_retry_decorator
# ---------------------------------------------------

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from ragas.run_config import RunConfig
from datasets import Dataset

import logging
logging.basicConfig(level=logging.INFO)

NUM_TEST_QUESTIONS = 50 

# State quản lý Key API
current_key_idx = 0
llm = None
embeddings = None

def init_llm(key_index):
    global llm, embeddings
    new_key = config.GEMINI_KEYS[key_index]
    print(f"\n[🔑] Đang chuyển sang API Key thứ {key_index + 1}/{len(config.GEMINI_KEYS)}...")
    
    # 1. Re-config thư viện google-generativeai gốc (cho generate_answer)
    genai.configure(api_key=new_key)
    
    # 2. Re-config biến môi trường (cho Ragas / Langchain)
    os.environ["GOOGLE_API_KEY"] = new_key
    
    # 3. Tạo lại instance LLM của Langchain
    # max_retries=0 để Langchain KHÔNG TỰ ĐỘNG RETRY ngầm khi vỡ Quota (vì ta muốn bắt lỗi 429 và đổi Key ngay)
    llm = ChatGoogleGenerativeAI(model=config.GEMINI_CHAT_MODEL, max_retries=0)
    # Tuy GoogleGenerativeAIEmbeddings không hỗ trợ max_retries qua biến kwargs ở một số phiên bản, 
    # nhưng nếu có lỗi nó sẽ bị văng ra catch luôn.
    embeddings = GoogleGenerativeAIEmbeddings(model=config.GEMINI_EMBEDDING_MODEL)

def next_key():
    global current_key_idx
    current_key_idx = (current_key_idx + 1) % len(config.GEMINI_KEYS)
    
    if current_key_idx == 0:
        print("\n[⚠️] TẤT CẢ CÁC KEY ĐÃ CẠN KIỆT TRONG NGÀY HÔM NAY!")
        print("Hệ thống sẽ ngủ đông 60 phút để hồi Key rồi chạy tiếp...")
        time.sleep(3600)
        
    init_llm(current_key_idx)

def run_evaluation():
    out_path = os.path.join(PROJECT_ROOT, "data", "Ragas_Final_Optimized_Results.csv")
    
    # 1. Cơ chế Checkpoint: Kiểm tra xem đã có file kết quả đang chạy dở chưa
    if os.path.exists(out_path):
        print(f"📁 Tìm thấy file {out_path}. Đang Resume (Chạy tiếp)...")
        df = pd.read_csv(out_path, encoding='utf-8-sig')
    else:
        # Nếu chưa có, đọc từ file gốc Test_Dataset
        csv_path = os.path.join(PROJECT_ROOT, "data", "Test_Dataset_Sinh12.csv")
        if not os.path.exists(csv_path):
            print("Không tìm thấy file Test_Dataset_Sinh12.csv!")
            return
                
        df = pd.read_csv(csv_path, encoding='utf-8-sig')
        # Bổ sung các cột trống để hứng kết quả
        df['answer'] = ""
        df['contexts'] = ""
        df['context_precision'] = pd.NA
        df['context_recall'] = pd.NA
        df['faithfulness'] = pd.NA
        df['answer_relevancy'] = pd.NA
        df['status'] = "PENDING"
    
    # Cắt gọn list theo số lượng test
    df = df.head(NUM_TEST_QUESTIONS)
    
    # Khởi tạo LLM với Key đầu tiên
    init_llm(current_key_idx)
    
    # 2. Xử lý từng câu (Row-by-Row)
    print(f"\n=== BẮT ĐẦU ĐÁNH GIÁ {len(df)} CÂU HỎI ===")
    
    for idx, row in df.iterrows():
        # Chỉ bỏ qua những câu đã DONE. Những câu bị ERROR trước đó sẽ được bắt chạy lại.
        if pd.notna(df.at[idx, 'status']) and df.at[idx, 'status'] == 'DONE':
            print(f"⏭️ Bỏ qua câu {idx+1}: Đã đánh giá xong trước đó.")
            continue
            
        q = row['question']
        ground_truth = row['ground_truth']
        print(f"\n[{idx+1}/{len(df)}] Đang xử lý: {q}")
        
        while True:
            try:
                import ast
                # A. Pha sinh câu trả lời (nếu chưa làm)
                if pd.isna(df.at[idx, 'answer']) or df.at[idx, 'answer'] == "":
                    q_emb = embed_query(q)
                    records = retrieve_context(q_emb, top_k=3, query_text=q)
                    ctxs = [rec["noi_dung"] for rec in records]
                    ans = generate_answer(q, records)
                    df.at[idx, 'answer'] = ans
                    df.at[idx, 'contexts'] = str(ctxs)
                    df.to_csv(out_path, index=False, encoding='utf-8-sig')
                    sys.consecutive_429 = 0
                    print("  ✅ Sinh câu trả lời thành công. Nghỉ 60 giây để reset 5 RPM...")
                    time.sleep(60)

                # Đọc lại contexts từ CSV (phòng trường hợp resume từ đầu)
                raw_ctxs = df.at[idx, 'contexts']
                if isinstance(raw_ctxs, str):
                    ctxs = ast.literal_eval(raw_ctxs)
                else:
                    ctxs = raw_ctxs

                # B. Pha chấm điểm Ragas (Từng tiêu chí một)
                data = {
                    "question": [q],
                    "answer": [df.at[idx, 'answer']],
                    "contexts": [ctxs],
                    "ground_truth": [ground_truth]
                }
                dataset = Dataset.from_dict(data)

                metrics_list = [
                    ('context_precision', context_precision),
                    ('context_recall', context_recall),
                    ('faithfulness', faithfulness),
                    ('answer_relevancy', answer_relevancy)
                ]

                for m_name, m_obj in metrics_list:
                    if pd.isna(df.at[idx, m_name]):
                        print(f"  -> Đang chấm tiêu chí: {m_name}...")
                        result = evaluate(
                            dataset=dataset,
                            metrics=[m_obj],
                            llm=llm,
                            embeddings=embeddings,
                            run_config=RunConfig(max_retries=0, max_wait=0, max_workers=1),
                            raise_exceptions=True
                        )
                        res_df = result.to_pandas()
                        df.at[idx, m_name] = res_df.at[0, m_name]
                        df.to_csv(out_path, index=False, encoding='utf-8-sig')
                        sys.consecutive_429 = 0
                        print(f"  ✅ Xong {m_name}. Ngủ 60s để reset Quota...")
                        time.sleep(60)
                
                # C. Cập nhật Checkpoint thẳng vào ổ cứng
                df.at[idx, 'status'] = 'DONE'
                df.to_csv(out_path, index=False, encoding='utf-8-sig')
                
                # TỰ ĐỘNG BACKUP RA EXCEL (.xlsx) ĐỂ CHỐNG LỖI HỎNG FONT!
                try:
                    df.to_excel(out_path.replace('.csv', '.xlsx'), index=False)
                except Exception:
                    pass
                
                print(f"🎉 Câu {idx+1} hoàn tất 100%!")
                break # Thoát khỏi vòng lặp While True để sang câu tiếp theo

                
            except Exception as e:
                err_str = str(e)
                err_type = type(e).__name__
                
                # Ragas đôi khi không trả về lỗi rõ GenerateRequestsPerMinute mà chỉ báo 429.
                # Nếu lập tức đổi key thì sẽ lãng phí 10 keys rất nhanh.
                if "429" in err_str or "quota" in err_str.lower() or "exhausted" in err_str.lower() or "limit" in err_str.lower():
                    if not hasattr(sys, 'consecutive_429'):
                        sys.consecutive_429 = 0
                    sys.consecutive_429 += 1
                    
                    if sys.consecutive_429 >= 3:
                        print(f"❌ Lỗi 429 liên tục 3 lần: Key hiện tại chắc chắn đã vỡ Quota Ngày. Đổi Key...")
                        next_key()
                        sys.consecutive_429 = 0
                    else:
                        print(f"⏳ Bị lỗi 429 (Lần {sys.consecutive_429}/3). Có thể do bị vướng RPM Limit, tạm nghỉ 65 giây rồi thử lại...")
                        time.sleep(65)
                elif "Timeout" in err_type or "Timeout" in err_str:
                    print(f"⏳ Lỗi Timeout (Kết nối mạng/API Google bị nghẽn). Tạm nghỉ 10 giây rồi thử lại...")
                    time.sleep(10)
                else:
                    print(f"❌ Lỗi không xác định ở câu {idx+1}: {repr(e)}")
                    time.sleep(10)
                    df.at[idx, 'status'] = 'ERROR'
                    df.to_csv(out_path, index=False, encoding='utf-8-sig')
                    break # Bỏ qua câu này để không bị kẹt vô hạn

    print("\n🎉 HOÀN TẤT TOÀN BỘ ĐÁNH GIÁ RAGAS!")

if __name__ == "__main__":
    run_evaluation()
