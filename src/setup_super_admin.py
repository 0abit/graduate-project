import firebase_admin
from firebase_admin import credentials, auth
import os

# Khởi tạo Firebase Admin (Nhớ trỏ đúng đường dẫn file JSON)
cred_path = "firebase-adminsdk.json"
if not os.path.exists(cred_path):
    print(f"Lỗi: Không tìm thấy file {cred_path}")
    exit(1)

cred = credentials.Certificate(cred_path)
firebase_admin.initialize_app(cred)

SUPER_ADMIN_EMAIL = "oabit666@gmail.com"

def set_super_admin(email):
    try:
        # 1. Tìm user bằng Email
        user = auth.get_user_by_email(email)
        
        # 2. Gắn Custom Claims: role = 'super_admin'
        # Đồng thời vẫn giữ lại quyền admin cũ (nếu có)
        current_claims = user.custom_claims or {}
        current_claims["role"] = "super_admin"
        current_claims["admin"] = True # Giữ lại cờ admin cho tương thích code cũ
        
        auth.set_custom_user_claims(user.uid, current_claims)
        
        print(f"✅ Thành công! Tài khoản {email} (UID: {user.uid}) đã trở thành SUPER ADMIN.")
        print("Lưu ý: User cần Đăng xuất và Đăng nhập lại trên web để nhận Token mới.")
        
    except auth.UserNotFoundError:
        print(f"❌ Lỗi: Không tìm thấy tài khoản với email {email} trên Firebase.")
    except Exception as e:
        print(f"❌ Lỗi không xác định: {e}")

if __name__ == "__main__":
    set_super_admin(SUPER_ADMIN_EMAIL)
