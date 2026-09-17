import os
import sys
import uuid
import datetime
from pathlib import Path
from typing import List, Optional

import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Request, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import cohere
from neo4j import GraphDatabase
from pydantic import BaseModel
from pymongo import MongoClient

import firebase_admin
from firebase_admin import credentials, auth

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src import config

# Initialize Firebase Admin
cred_path = os.path.join(PROJECT_ROOT, "firebase-adminsdk.json")
if os.path.exists(cred_path):
    cred = credentials.Certificate(cred_path)
    firebase_admin.initialize_app(cred)
else:
    print(f"WARNING: Firebase credentials not found at {cred_path}")

# Initialize MongoDB
mongo_client = MongoClient(config.MONGO_URI)
mongo_db = mongo_client[config.DB_NAME]
chat_history_col = mongo_db["chat_history"]

app = FastAPI(title="RAG Chatbot Sinh Học 12")

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Thư mục chứa giao diện tĩnh
STATIC_DIR = os.path.join(PROJECT_ROOT, "src", "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Cấu hình Neo4j
driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))

# Cấu hình Gemini
keys = config.GEMINI_KEYS
if not keys:
    raise ValueError("Chưa cấu hình GEMINI_KEYS")
genai.configure(api_key=keys[0])

security = HTTPBearer()

def verify_token(credentials: HTTPAuthorizationCredentials = Depends(security)):
    token = credentials.credentials
    try:
        # Cho phép sai số đồng hồ giữa máy tính và máy chủ Google là 10 giây
        decoded_token = auth.verify_id_token(token, clock_skew_seconds=10, check_revoked=True)
        return decoded_token
    except auth.RevokedIdTokenError:
        raise HTTPException(status_code=401, detail="TOKEN_REVOKED")
    except auth.UserDisabledError:
        raise HTTPException(status_code=401, detail="USER_DISABLED")
    except Exception as e:
        print(f"Firebase Auth Error: {e}")
        raise HTTPException(status_code=401, detail=f"Invalid authentication credentials: {e}")

def verify_admin_token(decoded_token: dict = Depends(verify_token)):
    if decoded_token.get('admin') is not True:
        raise HTTPException(status_code=403, detail="Admin privileges required")
    return decoded_token

# Models
class ChatRequest(BaseModel):
    query: str
    session_id: Optional[str] = None

class SourceDetail(BaseModel):
    bai_hoc: str
    noi_dung: str
    khai_niem: List[str]
    score: float
    tieu_muc: Optional[str] = None
    trang: Optional[List[int]] = None

class ChatResponse(BaseModel):
    answer: str
    sources: List[SourceDetail]
    session_id: str

def embed_query(query: str) -> List[float]:
    response = genai.embed_content(
        model=config.GEMINI_EMBEDDING_MODEL,
        content=query,
        task_type="retrieval_query"
    )
    return response['embedding']

import re as regex_module

def vector_search(query_embedding: List[float], top_k: int = 5) -> List[dict]:
    """Tìm kiếm bằng Vector (Cosine Similarity) - Hiểu ngữ nghĩa."""
    cypher = """
    CALL db.index.vector.queryNodes('PhanDoanV2_embedding', $top_k, $query_embedding)
    YIELD node, score
    MATCH (b:BaiHocV2)-[:CO_PHAN_DOAN]->(node)
    OPTIONAL MATCH (node)-[:DINH_NGHIA]->(kn:KhaiNiemV2)
    RETURN b.ten_hien_thi as bai_hoc, node.noi_dung as noi_dung, score,
           node.tieu_muc as tieu_muc, node.source_pages as trang,
           node.ma_phan_doan as chunk_id,
           collect(DISTINCT kn.ten) as khai_niem
    ORDER BY score DESC
    """
    with driver.session() as session:
        result = session.run(cypher, top_k=top_k, query_embedding=query_embedding)
        return [record.data() for record in result]

def escape_lucene_query(query: str) -> str:
    """Escape các ký tự đặc biệt của Lucene để tránh lỗi Full-text query."""
    # Danh sách ký tự đặc biệt của Lucene cần escape
    lucene_special = ['+', '-', '&', '|', '!', '(', ')', '{', '}',
                      '[', ']', '^', '"', '~', '*', '?', ':', '\\', '/']
    result = query
    for char in lucene_special:
        result = result.replace(char, '\\' + char)
    return result

def keyword_search(query: str, top_k: int = 5) -> List[dict]:
    """Tìm kiếm bằng từ khóa (Full-text/BM25) - Khớp chính xác ký tự."""
    safe_query = escape_lucene_query(query)
    cypher = """
    CALL db.index.fulltext.queryNodes('PhanDoanV2_fulltext', $search_text)
    YIELD node, score
    MATCH (b:BaiHocV2)-[:CO_PHAN_DOAN]->(node)
    OPTIONAL MATCH (node)-[:DINH_NGHIA]->(kn:KhaiNiemV2)
    RETURN b.ten_hien_thi as bai_hoc, node.noi_dung as noi_dung, score,
           node.tieu_muc as tieu_muc, node.source_pages as trang,
           node.ma_phan_doan as chunk_id,
           collect(DISTINCT kn.ten) as khai_niem
    ORDER BY score DESC
    LIMIT $top_k
    """
    with driver.session() as session:
        result = session.run(cypher, search_text=safe_query, top_k=top_k)
        return [record.data() for record in result]

def reciprocal_rank_fusion(vector_results: List[dict], keyword_results: List[dict], k: int = 60) -> List[dict]:
    """
    Thuật toán RRF (Cormack et al., 2009 - Đại học Waterloo).
    Gộp kết quả từ Vector Search và Keyword Search thành 1 danh sách duy nhất.
    Công thức: RRF(d) = Σ 1/(k + rank_r(d))
    """
    rrf_scores = {}      # chunk_id -> điểm RRF
    doc_map = {}         # chunk_id -> dữ liệu đầy đủ của tài liệu

    # Chấm điểm RRF cho kết quả từ Vector Search
    for rank, doc in enumerate(vector_results, start=1):
        chunk_id = doc.get('chunk_id', doc['noi_dung'][:50])
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank)
        doc_map[chunk_id] = doc

    # Chấm điểm RRF cho kết quả từ Keyword Search
    for rank, doc in enumerate(keyword_results, start=1):
        chunk_id = doc.get('chunk_id', doc['noi_dung'][:50])
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (k + rank)
        if chunk_id not in doc_map:
            doc_map[chunk_id] = doc

    # Sắp xếp theo điểm RRF giảm dần
    sorted_ids = sorted(rrf_scores.keys(), key=lambda x: rrf_scores[x], reverse=True)

    # Trả về danh sách đã xếp hạng, gán lại score = điểm RRF
    results = []
    for chunk_id in sorted_ids:
        doc = doc_map[chunk_id].copy()
        doc['score'] = round(rrf_scores[chunk_id], 6)
        results.append(doc)

    return results

def cohere_rerank(query: str, documents: List[dict], top_n: int = 3) -> List[dict]:
    """
    Sử dụng Cohere Rerank API để chấm lại điểm relevance cho các tài liệu.
    Input: danh sách tài liệu thô từ Hybrid Search.
    Output: top_n tài liệu được sắp xếp lại theo điểm Cohere.
    """
    cohere_api_key = os.getenv("COHERE_API_KEY", "")
    if not cohere_api_key or not documents:
        return documents[:top_n]
    
    try:
        co = cohere.ClientV2(api_key=cohere_api_key)
        
        # Chuẩn bị nội dung để Cohere chấm điểm
        doc_texts = [doc['noi_dung'] for doc in documents]
        
        response = co.rerank(
            model="rerank-v3.5",
            query=query,
            documents=doc_texts,
            top_n=top_n
        )
        
        # Sắp xếp lại danh sách theo kết quả Cohere
        reranked = []
        MIN_SCORE = 0.05  # Ngưỡng tối thiểu: loại bỏ tài liệu không liên quan
        for result in response.results:
            if result.relevance_score < MIN_SCORE:
                print(f"  [Cohere Rerank] Loại bỏ tài liệu index={result.index} (score={result.relevance_score:.4f} < {MIN_SCORE})")
                continue
            doc = documents[result.index].copy()
            doc['score'] = round(result.relevance_score, 6)
            reranked.append(doc)
        
        # Đảm bảo luôn có ít nhất 1 tài liệu (lấy tài liệu tốt nhất nếu tất cả bị lọc)
        if not reranked and response.results:
            best = response.results[0]
            doc = documents[best.index].copy()
            doc['score'] = round(best.relevance_score, 6)
            reranked.append(doc)
        
        print(f"  [Cohere Rerank] Reranked {len(documents)} -> {len(reranked)} documents (filtered by MIN_SCORE={MIN_SCORE})")
        return reranked
        
    except Exception as e:
        print(f"  [Cohere Rerank] Error: {e}. Fallback to RRF ranking.")
        return documents[:top_n]

def retrieve_context(query_embedding: List[float], top_k: int = 3, query_text: str = "") -> List[dict]:
    """
    Hybrid Search + Cohere Rerank:
    1. Vector Search + Keyword Search
    2. Gộp bằng RRF
    3. Lấy Top 10 ứng viên
    4. Cohere Rerank chấm lại -> Trả về Top K tốt nhất
    """
    # Bước 1: Vector Search (lấy dư để có nhiều ứng viên cho Rerank)
    fetch_k = max(top_k * 3, 10)
    vector_results = vector_search(query_embedding, top_k=fetch_k)

    # Bước 2: Keyword Search (chỉ chạy nếu có query_text)
    if query_text.strip():
        keyword_results = keyword_search(query_text, top_k=fetch_k)
    else:
        keyword_results = []

    # Bước 3: Gộp bằng RRF nếu có kết quả từ cả 2 nguồn
    if keyword_results:
        fused_results = reciprocal_rank_fusion(vector_results, keyword_results)
    else:
        fused_results = vector_results

    # Bước 4: Lấy Top 10 ứng viên từ RRF
    candidates = fused_results[:10]

    # Bước 5: Cohere Rerank chấm lại điểm, chỉ giữ Top K
    if query_text.strip():
        reranked = cohere_rerank(query_text, candidates, top_n=top_k)
    else:
        reranked = candidates[:top_k]

    return reranked

def generate_answer(query: str, context_records: List[dict]) -> str:
    if not context_records:
        return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong sách giáo khoa Sinh Học 12."
        
    context_text = ""
    for idx, record in enumerate(context_records, 1):
        context_text += f"--- Nguồn {idx} (Bài: {record['bai_hoc']}) ---\n"
        context_text += f"{record['noi_dung']}\n\n"
        
    prompt = f"""BẠN LÀ TRỢ LÝ HỌC TẬP SINH HỌC 12. Bạn CHỈ được phép trả lời dựa trên TÀI LIỆU bên dưới.

═══ QUY TẮC BẮT BUỘC ═══

1. ĐỌC KỸ tài liệu trước. Xác định chính xác đoạn nào chứa câu trả lời.
2. CHỈ SỬ DỤNG thông tin có trong tài liệu. Mỗi câu trong câu trả lời phải truy nguyên được về một đoạn cụ thể trong tài liệu.
3. KHÔNG ĐƯỢC: bịa thêm ví dụ, thêm giải thích mở rộng, thêm kiến thức ngoài, thêm câu tổng kết/kết luận mà tài liệu không đề cập.
4. Nếu tài liệu KHÔNG CHỨA câu trả lời → Nói: "Thông tin này không có trong tài liệu em đang tham khảo."
5. Nếu tài liệu CHỈ CHỨA MỘT PHẦN câu trả lời → Trả lời phần có trong tài liệu, rồi nói rõ: "Các nội dung khác không được đề cập trong tài liệu này."
6. Trả lời ĐÚNG trọng tâm câu hỏi, KHÔNG lan man sang chủ đề khác kể cả khi chúng liên quan.
7. Diễn đạt tự nhiên, thân thiện. Dùng "Theo sách giáo khoa Sinh học 12..." thay vì "Theo Nguồn 1".

═══ VÍ DỤ MẪU ═══

Câu hỏi: "Đột biến gen là gì?"
Tài liệu: "Đột biến gen là những biến đổi trong cấu trúc của gen, liên quan đến một hoặc một số cặp nucleotide."
Trả lời đúng: "Theo sách giáo khoa Sinh học 12, đột biến gen là những biến đổi trong cấu trúc của gen, liên quan đến một hoặc một số cặp nucleotide."
Trả lời SAI (bịa thêm): "Đột biến gen là những biến đổi trong cấu trúc của gen... Ví dụ như bệnh hồng cầu hình liềm là do đột biến thay thế cặp A-T bằng T-A..." ← SAI vì thông tin về bệnh hồng cầu hình liềm không có trong tài liệu được cung cấp.

═══ TÀI LIỆU ═══
{context_text}
═══ CÂU HỎI ═══
{query}

═══ TRẢ LỜI (chỉ dựa trên tài liệu ở trên) ═══
"""
    model = genai.GenerativeModel(
        model_name=config.GEMINI_CHAT_MODEL,
        generation_config=genai.GenerationConfig(
            temperature=config.CHATBOT_TEMPERATURE
        )
    )
    response = model.generate_content(prompt)
    return response.text

import json

def verify_answer(query: str, context_records: List[dict], answer: str) -> dict:
    """
    Bước kiểm tra chất lượng câu trả lời (Post-Generation Verification).
    Sử dụng Gemini Flash với vai trò Verifier để kiểm tra:
    - Faithfulness: câu trả lời có bịa thêm nội dung ngoài tài liệu không?
    - Relevancy: câu trả lời có đúng trọng tâm câu hỏi không?
    
    Returns dict: {"verdict": "PASS"/"FAIL", "is_faithful": bool, "is_relevant": bool, "issues": [...]}
    """
    if not config.ENABLE_ANSWER_VERIFY:
        return {"verdict": "PASS", "is_faithful": True, "is_relevant": True, "issues": [], "skipped": True}
    
    if not answer or not context_records:
        return {"verdict": "PASS", "is_faithful": True, "is_relevant": True, "issues": [], "skipped": True}
    
    # Chuẩn bị context text cho Verifier
    context_text = ""
    for idx, record in enumerate(context_records, 1):
        context_text += f"--- Nguồn {idx} (Bài: {record['bai_hoc']}) ---\n"
        context_text += f"{record['noi_dung']}\n\n"
    
    verify_prompt = f"""BẠN LÀ BỘ KIỂM TRA CHẤT LƯỢNG câu trả lời cho hệ thống hỏi đáp Sinh Học 12.

═══ NHIỆM VỤ ═══
Kiểm tra xem CÂU TRẢ LỜI bên dưới có vi phạm quy tắc nào không.

═══ QUY TẮC KIỂM TRA ═══
1. FAITHFULNESS: Mỗi câu khẳng định trong CÂU TRẢ LỜI phải có cơ sở trong TÀI LIỆU. Nếu câu trả lời chứa ví dụ, số liệu, hoặc giải thích KHÔNG có trong tài liệu → đánh dấu is_faithful = false.
2. RELEVANCY: CÂU TRẢ LỜI phải đúng trọng tâm CÂU HỎI, không lan man sang chủ đề khác.
3. Nếu câu trả lời là từ chối hợp lệ (ví dụ: "Thông tin này không có trong tài liệu...") → đó là PASS.
4. Các cụm từ chuyển tiếp tự nhiên như "Theo sách giáo khoa Sinh học 12..." KHÔNG phải vi phạm.

═══ TRẢ VỀ JSON (chỉ JSON, không thêm text) ═══
{{
  "is_faithful": true hoặc false,
  "is_relevant": true hoặc false,
  "issues": ["mô tả ngắn gọn từng vấn đề nếu có, mảng rỗng nếu không có"],
  "verdict": "PASS" hoặc "FAIL"
}}

═══ TÀI LIỆU ═══
{context_text}
═══ CÂU HỎI ═══
{query}

═══ CÂU TRẢ LỜI CẦN KIỂM TRA ═══
{answer}
"""
    
    try:
        model = genai.GenerativeModel(
            model_name=config.GEMINI_CHAT_MODEL,
            generation_config=genai.GenerationConfig(
                temperature=config.VERIFY_TEMPERATURE,
                response_mime_type="application/json"
            )
        )
        response = model.generate_content(verify_prompt)
        result = json.loads(response.text)
        
        # Đảm bảo các trường bắt buộc tồn tại
        result.setdefault("is_faithful", True)
        result.setdefault("is_relevant", True)
        result.setdefault("issues", [])
        result.setdefault("verdict", "PASS" if result["is_faithful"] and result["is_relevant"] else "FAIL")
        
        print(f"  [Verify] verdict={result['verdict']}, faithful={result['is_faithful']}, relevant={result['is_relevant']}, issues={result['issues']}")
        return result
        
    except Exception as e:
        # Fail-open: nếu verify lỗi thì vẫn trả kết quả cho user
        print(f"  [Verify] Error: {e}. Skipping verification (fail-open).")
        return {"verdict": "PASS", "is_faithful": True, "is_relevant": True, "issues": [], "error": str(e)}

def regenerate_answer_strict(query: str, context_records: List[dict], issues: list) -> str:
    """
    Sinh lại câu trả lời với prompt nghiêm ngặt hơn khi verify phát hiện vấn đề.
    Thêm cảnh báo cụ thể về các lỗi đã phát hiện.
    """
    if not context_records:
        return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong sách giáo khoa Sinh Học 12."
    
    context_text = ""
    for idx, record in enumerate(context_records, 1):
        context_text += f"--- Nguồn {idx} (Bài: {record['bai_hoc']}) ---\n"
        context_text += f"{record['noi_dung']}\n\n"
    
    issues_text = "\n".join(f"- {issue}" for issue in issues) if issues else "Không xác định cụ thể."
    
    strict_prompt = f"""BẠN LÀ TRỢ LÝ HỌC TẬP SINH HỌC 12. Bạn CHỈ được phép trả lời dựa trên TÀI LIỆU bên dưới.

⚠️ CẢNH BÁO: Câu trả lời trước đó đã bị phát hiện có vấn đề:
{issues_text}

═══ QUY TẮC BẮT BUỘC (NGHIÊM NGẶT) ═══

1. ĐỌC KỸ tài liệu trước. Xác định chính xác đoạn nào chứa câu trả lời.
2. CHỈ SỬ DỤNG thông tin có trong tài liệu. Mỗi câu trong câu trả lời phải truy nguyên được về một đoạn cụ thể.
3. TUYỆT ĐỐI KHÔNG ĐƯỢC: bịa thêm ví dụ, thêm giải thích mở rộng, thêm kiến thức ngoài, thêm câu tổng kết/kết luận mà tài liệu không đề cập.
4. Nếu tài liệu KHÔNG CHỨA câu trả lời → Nói: "Thông tin này không có trong tài liệu em đang tham khảo."
5. Nếu tài liệu CHỈ CHỨA MỘT PHẦN câu trả lời → Trả lời phần có trong tài liệu, rồi nói rõ: "Các nội dung khác không được đề cập trong tài liệu này."
6. Trả lời ĐÚNG trọng tâm câu hỏi, KHÔNG lan man.
7. Diễn đạt tự nhiên, thân thiện. Dùng "Theo sách giáo khoa Sinh học 12..." thay vì "Theo Nguồn 1".

═══ TÀI LIỆU ═══
{context_text}
═══ CÂU HỎI ═══
{query}

═══ TRẢ LỜI (chỉ dựa trên tài liệu, sửa các lỗi đã nêu) ═══
"""
    
    model = genai.GenerativeModel(
        model_name=config.GEMINI_CHAT_MODEL,
        generation_config=genai.GenerationConfig(
            temperature=0.0  # Deterministic cho lần retry
        )
    )
    response = model.generate_content(strict_prompt)
    return response.text


@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

import time
user_last_request = {}
COOLDOWN_SECONDS = 5

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, decoded_token: dict = Depends(verify_token)):
    user_id = decoded_token['uid']
    
    current_time = time.time()
    if user_id in user_last_request:
        time_since_last = current_time - user_last_request[user_id]
        if time_since_last < COOLDOWN_SECONDS:
            raise HTTPException(status_code=429, detail=f"Vui lòng đợi {int(COOLDOWN_SECONDS - time_since_last)} giây trước khi gửi câu hỏi tiếp theo.")
    user_last_request[user_id] = current_time

    try:
        query_embedding = embed_query(request.query)
        context_records = retrieve_context(query_embedding, top_k=3, query_text=request.query)
        answer = generate_answer(request.query, context_records)
        
        # Bước kiểm tra chất lượng câu trả lời (Post-Generation Verification)
        verification = verify_answer(request.query, context_records, answer)
        
        if verification.get("verdict") == "FAIL":
            print(f"  [Verify] FAIL detected. Regenerating with stricter prompt...")
            answer = regenerate_answer_strict(request.query, context_records, verification.get("issues", []))
            # Verify lần 2 (không retry thêm nữa để tránh loop)
            verification_retry = verify_answer(request.query, context_records, answer)
            verification = verification_retry  # Cập nhật kết quả verify cuối cùng
        
        sources = [
            SourceDetail(
                bai_hoc=rec["bai_hoc"],
                noi_dung=rec["noi_dung"],
                khai_niem=rec["khai_niem"],
                score=rec["score"],
                tieu_muc=rec.get("tieu_muc"),
                trang=rec.get("trang")
            )
            for rec in context_records
        ]
        
        session_id = request.session_id
        if not session_id:
            session_id = str(uuid.uuid4())
            # Create new session
            chat_history_col.insert_one({
                "session_id": session_id,
                "user_id": user_id,
                "title": request.query[:50] + "..." if len(request.query) > 50 else request.query,
                "created_at": datetime.datetime.utcnow(),
                "messages": []
            })
            
        # Append messages (bao gồm verification metadata)
        chat_history_col.update_one(
            {"session_id": session_id, "user_id": user_id},
            {"$push": {"messages": {
                "$each": [
                    {"role": "user", "content": request.query, "timestamp": datetime.datetime.utcnow()},
                    {"role": "bot", "content": answer, "sources": [s.dict() for s in sources],
                     "verification": verification, "timestamp": datetime.datetime.utcnow()}
                ]
            }}}
        )
        
        return ChatResponse(answer=answer, sources=sources, session_id=session_id)
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/history")
def get_history(decoded_token: dict = Depends(verify_token)):
    user_id = decoded_token['uid']
    histories = chat_history_col.find({"user_id": user_id}, {"_id": 0, "messages": 0}).sort("created_at", -1)
    return list(histories)

@app.get("/history/{session_id}")
def get_session(session_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = decoded_token['uid']
    session = chat_history_col.find_one({"session_id": session_id, "user_id": user_id}, {"_id": 0})
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return session

@app.delete("/history/{session_id}")
def delete_session(session_id: str, decoded_token: dict = Depends(verify_token)):
    user_id = decoded_token['uid']
    result = chat_history_col.delete_one({"session_id": session_id, "user_id": user_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"status": "success"}

# --- ADMIN APIs ---

@app.get("/auth/me")
def get_me(decoded_token: dict = Depends(verify_token)):
    email = decoded_token.get('email', '')
    uid = decoded_token['uid']
    
    # Auto-grant admin if email matches SUPER_ADMIN_EMAIL
    if email.lower() == config.SUPER_ADMIN_EMAIL.lower() and decoded_token.get('admin') is not True:
        auth.set_custom_user_claims(uid, {'admin': True})
        return {"role": "admin"}
        
    role = "admin" if decoded_token.get('admin') is True else "user"
    return {"role": role, "email": email}

@app.get("/admin/users")
def get_all_users(admin_token: dict = Depends(verify_admin_token)):
    try:
        users = []
        # Tạm thời fetch 1000 users, nếu nhiều hơn phải phân trang (thường ít dùng ở project nhỏ)
        page = auth.list_users(max_results=1000)
        for user in page.users:
            users.append({
                "uid": user.uid,
                "email": user.email,
                "disabled": user.disabled,
                "admin": user.custom_claims.get('admin', False) if user.custom_claims else False
            })
        return users
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class RoleRequest(BaseModel):
    is_admin: bool

@app.post("/admin/users/{uid}/role")
def update_user_role(uid: str, request: RoleRequest, admin_token: dict = Depends(verify_admin_token)):
    try:
        # --- LUẬT BẢO VỆ SUPER ADMIN ---
        target_user = auth.get_user(uid)
        target_claims = target_user.custom_claims or {}
        if target_claims.get("role") == "super_admin" or target_user.email == "oabit666@gmail.com":
            raise HTTPException(status_code=403, detail="Bạn không thể sửa quyền của Super Admin!")
            
        auth.set_custom_user_claims(uid, {'admin': request.is_admin})
        return {"status": "success", "admin": request.is_admin}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class StatusRequest(BaseModel):
    disabled: bool

@app.post("/admin/users/{uid}/toggle-status")
def toggle_user_status(uid: str, request: StatusRequest, admin_token: dict = Depends(verify_admin_token)):
    try:
        # --- LUẬT BẢO VỆ SUPER ADMIN ---
        if request.disabled: # Chỉ chặn nếu hành động là "Khóa"
            target_user = auth.get_user(uid)
            target_claims = target_user.custom_claims or {}
            if target_claims.get("role") == "super_admin" or target_user.email == "oabit666@gmail.com":
                raise HTTPException(status_code=403, detail="Bạn không thể khóa tài khoản của Super Admin!")
                
        auth.update_user(uid, disabled=request.disabled)
        return {"status": "success", "disabled": request.disabled}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/admin/users/{uid}/reset-password")
def reset_user_password(uid: str, admin_token: dict = Depends(verify_admin_token)):
    try:
        user = auth.get_user(uid)
        if not user.email:
            raise HTTPException(status_code=400, detail="User has no email")
        # Gửi email qua Firebase
        link = auth.generate_password_reset_link(user.email)
        # Trong môi trường thực tế, nếu không có SMTP server riêng, Firebase client SDK 
        # sendPasswordResetEmail() phía frontend sẽ tiện hơn.
        # Ở đây ta trả về link cho admin, hoặc admin tự gửi qua một email server.
        # Lưu ý: Firebase không có hàm nào để trigger gửi email *trực tiếp* từ Admin SDK, nó chỉ generate link.
        return {"status": "success", "reset_link": link}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="localhost", port=8000)
