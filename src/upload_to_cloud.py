import os
import re
import sys
import subprocess
from pathlib import Path

# Ensure cloudinary is installed
try:
    import cloudinary
    import cloudinary.uploader
    from cloudinary.utils import cloudinary_url
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cloudinary"])
    import cloudinary
    import cloudinary.uploader
    from cloudinary.utils import cloudinary_url

from neo4j import GraphDatabase
import config

if not config.CLOUDINARY_URL:
    print("❌ Lỗi: Không tìm thấy CLOUDINARY_URL trong config.")
    sys.exit(1)

match = re.match(r"cloudinary://([^:]+):([^@]+)@(.+)", config.CLOUDINARY_URL)
if match:
    api_key, api_secret, cloud_name = match.groups()
    cloudinary.config(
        cloud_name=cloud_name,
        api_key=api_key,
        api_secret=api_secret,
        secure=True
    )
else:
    print("❌ Lỗi: Định dạng CLOUDINARY_URL không hợp lệ.")
    sys.exit(1)

# 1. Upload local images to Cloudinary
extracted_dir = Path(config.EXTRACTED_IMAGE_DIR) / config.CURRENT_BOOK
url_mapping = {}

print("☁️ Đang tải ảnh lên Cloudinary...")
if extracted_dir.exists():
    for img_path in extracted_dir.glob("*.png"): # phase1 saves as .png
        local_ref = f"/static/extracted_images/{config.CURRENT_BOOK}/{img_path.name}"
        print(f"Uploading {img_path.name}...")
        
        # Upload
        upload_result = cloudinary.uploader.upload(
            str(img_path),
            folder=f"bioai_rag/{config.CURRENT_BOOK}",
            public_id=img_path.stem
        )
        secure_url = upload_result.get("secure_url")
        url_mapping[local_ref] = secure_url
        print(f"  -> {secure_url}")
else:
    print("Thư mục ảnh không tồn tại. Bỏ qua bước upload.")

if not url_mapping:
    print("Không tìm thấy ảnh nào cần upload.")
    sys.exit(0)

# 2. Update Neo4j
print("\n🔄 Đang cập nhật Database Neo4j...")
try:
    driver = GraphDatabase.driver(config.NEO4J_URI, auth=(config.NEO4J_USER, config.NEO4J_PASS))
    with driver.session() as session:
        for local_ref, secure_url in url_mapping.items():
            query = """
            MATCH (n)
            WHERE n.noi_dung CONTAINS $local_ref
            SET n.noi_dung = replace(n.noi_dung, $local_ref, $secure_url)
            RETURN count(n) as updated_count
            """
            result = session.run(query, local_ref=local_ref, secure_url=secure_url)
            count = result.single()["updated_count"]
            if count > 0:
                print(f"  ✅ Cập nhật {count} nodes chứa {img_path.name}")
    driver.close()
except Exception as e:
    print(f"⚠️ Lỗi cập nhật Neo4j: {e}")

# 3. Update Markdown files
print("\n📝 Đang cập nhật file Markdown...")
md_file = Path(config.MD_DIR) / f"{config.CURRENT_BOOK}.md"
if md_file.exists():
    content = md_file.read_text(encoding="utf-8")
    for local_ref, secure_url in url_mapping.items():
        content = content.replace(local_ref, secure_url)
    md_file.write_text(content, encoding="utf-8")
    print(f"  ✅ Đã cập nhật xong file {md_file.name}")
else:
    print(f"⚠️ Không tìm thấy file {md_file.name}")

print("\n🎉 HOÀN TẤT! Bạn có thể xóa thư mục ảnh cục bộ.")
