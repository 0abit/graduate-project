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
from neo4j import GraphDatabase
from pydantic import BaseModel
from pymongo import MongoClient

import firebase_admin
from firebase_admin import credentials, auth

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import config

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
        decoded_token = auth.verify_id_token(token, clock_skew_seconds=10)
        return decoded_token
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

def retrieve_context(query_embedding: List[float], top_k: int = 3) -> List[dict]:
    query = """
    CALL db.index.vector.queryNodes('PhanDoanV2_embedding', $top_k, $query_embedding)
    YIELD node, score
    MATCH (b:BaiHocV2)-[:CO_PHAN_DOAN]->(node)
    OPTIONAL MATCH (node)-[:DINH_NGHIA]->(kn:KhaiNiemV2)
    RETURN b.ten_hien_thi as bai_hoc, node.noi_dung as noi_dung, score,
           node.muc as tieu_muc, node.source_pages as trang,
           collect(DISTINCT kn.ten) as khai_niem
    ORDER BY score DESC
    """
    
    with driver.session() as session:
        result = session.run(query, top_k=top_k, query_embedding=query_embedding)
        return [record.data() for record in result]

def generate_answer(query: str, context_records: List[dict]) -> str:
    if not context_records:
        return "Xin lỗi, tôi không tìm thấy thông tin liên quan trong sách giáo khoa Sinh Học 12."
        
    context_text = ""
    for idx, record in enumerate(context_records, 1):
        context_text += f"--- Nguồn {idx} (Bài: {record['bai_hoc']}) ---\n"
        context_text += f"{record['noi_dung']}\n\n"
        
    prompt = f"""
Bạn là một trợ lý ảo chuyên môn về Sinh Học lớp 12. Nhiệm vụ của bạn là trả lời câu hỏi của người dùng dựa trên các tài liệu trích xuất từ sách giáo khoa dưới đây.
Hãy trả lời một cách chính xác, thân thiện, dễ hiểu và CHỈ sử dụng thông tin từ tài liệu được cung cấp. Nếu thông tin không có trong tài liệu, hãy nói rõ là bạn không tìm thấy.
Không bịa đặt thêm thông tin ngoài tài liệu.
NẾU tài liệu bạn SỬ DỤNG ĐỂ TRẢ LỜI có chứa các thẻ hình ảnh (ví dụ: `![Hình ảnh minh hoạ](/static/...)` hoặc link Cloudinary), hãy trích dẫn nguyên vẹn thẻ hình ảnh đó vào câu trả lời để minh hoạ. TUYỆT ĐỐI KHÔNG chèn hình ảnh từ các đoạn tài liệu không liên quan đến câu trả lời.
QUAN TRỌNG: Khi trả lời, KHÔNG sử dụng các cụm từ máy móc như "Theo Nguồn 1", "Dựa vào Nguồn 2". Thay vào đó, hãy diễn đạt tự nhiên như "Theo sách giáo khoa Sinh học 12...", hoặc nhắc tên bài học.

TÀI LIỆU:
{context_text}

CÂU HỎI CỦA NGƯỜI DÙNG:
{query}

TRẢ LỜI:
"""
    model = genai.GenerativeModel(
        model_name=config.GEMINI_CHAT_MODEL,
        generation_config=genai.GenerationConfig(
            temperature=config.CHATBOT_TEMPERATURE
        )
    )
    response = model.generate_content(prompt)
    return response.text

@app.get("/")
def serve_frontend():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))

@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest, decoded_token: dict = Depends(verify_token)):
    user_id = decoded_token['uid']
    try:
        query_embedding = embed_query(request.query)
        context_records = retrieve_context(query_embedding, top_k=3)
        answer = generate_answer(request.query, context_records)
        
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
            
        # Append messages
        chat_history_col.update_one(
            {"session_id": session_id, "user_id": user_id},
            {"$push": {"messages": {
                "$each": [
                    {"role": "user", "content": request.query, "timestamp": datetime.datetime.utcnow()},
                    {"role": "bot", "content": answer, "sources": [s.dict() for s in sources], "timestamp": datetime.datetime.utcnow()}
                ]
            }}}
        )
        
        return ChatResponse(answer=answer, sources=sources, session_id=session_id)
        
    except Exception as e:
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
        auth.set_custom_user_claims(uid, {'admin': request.is_admin})
        return {"status": "success", "admin": request.is_admin}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

class StatusRequest(BaseModel):
    disabled: bool

@app.post("/admin/users/{uid}/toggle-status")
def toggle_user_status(uid: str, request: StatusRequest, admin_token: dict = Depends(verify_admin_token)):
    try:
        auth.update_user(uid, disabled=request.disabled)
        return {"status": "success", "disabled": request.disabled}
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
