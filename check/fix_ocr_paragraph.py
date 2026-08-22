"""Sửa đoạn văn sai về PDF ở Chương 2."""
from docx import Document
import copy

filepath = r'd:\OABIT\do an sinh hoc\viết báo cáo\Chuong_2_Tong_Quan_Ly_Thuyet.docx'
doc = Document(filepath)

OLD_TEXT = (
    "Dữ liệu đầu vào của đồ án là file PDF sách giáo khoa. "
    "PDF không phải văn bản mà nó là ảnh. "
    "Máy tính không đọc được chữ trên ảnh như con người. "
    "Để biến hàng trăm trang sách in thành dữ liệu văn bản mà hệ thống RAG có thể xử lý, "
    "bước đầu tiên bắt buộc phải giải quyết là Nhận dạng ký tự quang học "
    "(Optical Character Recognition - OCR). "
    "Công nghệ này nhận đầu vào là hình ảnh chứa chữ (trang scan, ảnh chụp), "
    "và trả ra chuỗi ký tự tương ứng dưới dạng text thuần. "
    "Trong đồ án, OCR là cổng vào của toàn bộ đường ống dữ liệu "
    "và nếu bước này sai, mọi bước sau đều sai theo."
)

NEW_TEXT = (
    "Dữ liệu đầu vào của đồ án là file PDF sách giáo khoa. "
    "PDF tồn tại ở hai dạng: PDF văn bản (text-based) — máy tính đọc được chữ trực tiếp vì dữ liệu ký tự được mã hóa sẵn bên trong file; "
    "và PDF dạng scan (image-based) — mỗi trang chỉ là một bức ảnh chụp hoặc quét từ sách in, máy tính không thể trích xuất chữ nếu không có công cụ chuyên biệt. "
    "File SGK Sinh học 12 sử dụng trong đồ án thuộc dạng scan. "
    "Máy tính nhìn vào mỗi trang chỉ thấy một ma trận pixel, không thấy ký tự nào cả. "
    "Để biến hàng trăm trang sách in thành dữ liệu văn bản mà hệ thống RAG có thể xử lý, "
    "bước đầu tiên bắt buộc phải giải quyết là Nhận dạng ký tự quang học "
    "(Optical Character Recognition — OCR). "
    "Công nghệ này nhận đầu vào là hình ảnh chứa chữ (trang scan, ảnh chụp), "
    "và trả ra chuỗi ký tự tương ứng dưới dạng text thuần. "
    "Trong đồ án, OCR là cổng vào của toàn bộ đường ống dữ liệu — nếu bước này sai, mọi bước sau đều sai theo."
)

found = False
for i, para in enumerate(doc.paragraphs):
    if "PDF không phải văn bản mà nó là ảnh" in para.text:
        print(f"Tìm thấy đoạn cần sửa tại paragraph {i}")
        print(f"NỘI DUNG CŨ:\n{para.text}\n")
        
        # Giữ nguyên format (font, size, bold...) của run đầu tiên
        if para.runs:
            original_run = para.runs[0]
            font_name = original_run.font.name
            font_size = original_run.font.size
            font_bold = original_run.font.bold
            font_italic = original_run.font.italic
        
        # Xóa toàn bộ run cũ
        for run in para.runs:
            run.text = ""
        
        # Ghi nội dung mới vào run đầu tiên
        para.runs[0].text = NEW_TEXT
        
        # Khôi phục format
        if font_name:
            para.runs[0].font.name = font_name
        if font_size:
            para.runs[0].font.size = font_size
        if font_bold is not None:
            para.runs[0].font.bold = font_bold
        if font_italic is not None:
            para.runs[0].font.italic = font_italic
        
        print(f"NỘI DUNG MỚI:\n{para.text}\n")
        found = True
        break

if found:
    doc.save(filepath)
    print("✅ Đã lưu file thành công!")
else:
    print("❌ Không tìm thấy đoạn văn cần sửa.")
