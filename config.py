import os
from dotenv import load_dotenv
from toc_mappings.sinh_hoc_12 import TOC_MAPPING as SINH_HOC_12_TOC_MAPPING

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "data"))
PDF_DIR = os.getenv("PDF_DIR", os.path.join(DATA_DIR, "raw_data"))
MD_DIR = os.getenv("MD_DIR", os.path.join(DATA_DIR, "processed_md"))
OCR_DIR = os.getenv("OCR_DIR", os.path.join(DATA_DIR, "ocr"))
OCR_IMAGE_DIR = os.getenv("OCR_IMAGE_DIR", os.path.join(OCR_DIR, "images"))
OCR_JSON_DIR = os.getenv("OCR_JSON_DIR", os.path.join(OCR_DIR, "json"))
METADATA_DIR = os.getenv("METADATA_DIR", os.path.join(DATA_DIR, "metadata"))
CHECKPOINT_DIR = os.getenv("CHECKPOINT_DIR", os.path.join(DATA_DIR, "checkpoints"))
STOP_FILE = os.getenv("STOP_FILE", os.path.join(DATA_DIR, "STOP_PIPELINE"))

os.makedirs(PDF_DIR, exist_ok=True)
os.makedirs(MD_DIR, exist_ok=True)
os.makedirs(OCR_IMAGE_DIR, exist_ok=True)
os.makedirs(OCR_JSON_DIR, exist_ok=True)
os.makedirs(METADATA_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017/")
DB_NAME = os.getenv("DB_NAME", "nlp_biology_db")
COLLECTION_NAME = "markdown_chunks"
COLLECTION_NAME_V2 = "markdown_chunks_v2"
EXTRACTED_IMAGE_DIR = os.getenv("EXTRACTED_IMAGE_DIR", os.path.join(BASE_DIR, "src", "static", "extracted_images"))
os.makedirs(EXTRACTED_IMAGE_DIR, exist_ok=True)

NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASS = os.getenv("NEO4J_PASS")

CLOUDINARY_URL = os.getenv("CLOUDINARY_URL")
SUPER_ADMIN_EMAIL = os.getenv("SUPER_ADMIN_EMAIL", "oabit666@gmail.com")

GEMINI_KEYS = [k.strip() for k in os.getenv("GEMINI_KEYS", "").split(",") if k.strip()]

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "3"))
SLEEP_TIME = int(os.getenv("SLEEP_TIME", "300"))
KEY_SLEEP_TIME = int(os.getenv("KEY_SLEEP_TIME", "180"))
PHASE3_KEY_GROUP_SIZE = int(os.getenv("PHASE3_KEY_GROUP_SIZE", str(BATCH_SIZE)))
PHASE3_GROUP_SLEEP_TIME = int(os.getenv("PHASE3_GROUP_SLEEP_TIME", str(KEY_SLEEP_TIME)))
PHASE3_ALL_KEYS_EXHAUSTED_SLEEP_TIME = int(os.getenv("PHASE3_ALL_KEYS_EXHAUSTED_SLEEP_TIME", "3600"))
PHASE3_JSON_RETRY_COUNT = int(os.getenv("PHASE3_JSON_RETRY_COUNT", "2"))
PHASE3_JSON_RETRY_SLEEP_TIME = int(os.getenv("PHASE3_JSON_RETRY_SLEEP_TIME", "5"))
PHASE3_TRANSIENT_RETRY_COUNT = int(os.getenv("PHASE3_TRANSIENT_RETRY_COUNT", "3"))
PHASE3_TRANSIENT_RETRY_SLEEP_TIME = int(os.getenv("PHASE3_TRANSIENT_RETRY_SLEEP_TIME", "30"))
PHASE3_SKIP_CHUNK_ON_JSON_ERROR = os.getenv("PHASE3_SKIP_CHUNK_ON_JSON_ERROR", "true").lower() in {"1", "true", "yes", "y"}
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3-flash-preview")
GEMINI_EMBEDDING_MODEL = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-2")
GEMINI_CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3-flash-preview")
CHATBOT_TEMPERATURE = float(os.getenv("CHATBOT_TEMPERATURE", "0.1"))

PHASE5_BATCH_SIZE = int(os.getenv("PHASE5_BATCH_SIZE", "3"))
PHASE5_SLEEP_TIME = int(os.getenv("PHASE5_SLEEP_TIME", "180"))
PHASE5_ERROR_SLEEP_TIME = int(os.getenv("PHASE5_ERROR_SLEEP_TIME", "180"))

CURRENT_BOOK = os.getenv("CURRENT_BOOK", "sinh_hoc_12")

PHASE1_OCR_MODE = os.getenv("PHASE1_OCR_MODE", "paddle_vietocr").lower()

PADDLE_OCR_LANG = os.getenv("PADDLE_OCR_LANG", "vi")
PADDLE_OCR_VERSION = os.getenv("PADDLE_OCR_VERSION", "PP-OCRv4")
PADDLE_DEVICE = os.getenv("PADDLE_DEVICE", "cpu")
PADDLE_USE_TEXTLINE_ORIENTATION = os.getenv("PADDLE_USE_TEXTLINE_ORIENTATION", "true").lower() in {"1", "true", "yes", "y"}
PADDLE_PDF_DPI = int(os.getenv("PADDLE_PDF_DPI", "220"))
PHASE1_FORCE_REOCR = os.getenv("PHASE1_FORCE_REOCR", "false").lower() in {"1", "true", "yes", "y"}
PADDLE_ENABLE_MKLDNN = os.getenv("PADDLE_ENABLE_MKLDNN", "false").lower() in {"1", "true", "yes", "y"}

VIETOCR_CONFIG = os.getenv("VIETOCR_CONFIG", "vgg_transformer")
VIETOCR_DEVICE = os.getenv("VIETOCR_DEVICE", "cpu")
VIETOCR_BEAMSEARCH = os.getenv("VIETOCR_BEAMSEARCH", "false").lower() in {"1", "true", "yes", "y"}
VIETOCR_CROP_PADDING = int(os.getenv("VIETOCR_CROP_PADDING", "4"))
PHASE1_KEEP_PADDLE_TEXT = os.getenv("PHASE1_KEEP_PADDLE_TEXT", "false").lower() in {"1", "true", "yes", "y"}

CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "3500"))
INCLUDE_OUT_OF_TOC_PAGES = os.getenv("INCLUDE_OUT_OF_TOC_PAGES", "false").lower() in {"1", "true", "yes", "y"}
OBJECTIVE_SCAN_PAGES = int(os.getenv("OBJECTIVE_SCAN_PAGES", "2"))
PHASE2_FORCE_METADATA_REEXTRACT = os.getenv("PHASE2_FORCE_METADATA_REEXTRACT", "true").lower() in {"1", "true", "yes", "y"}

BOOK_TOC_MAPPINGS = {
    "sinh_hoc_12": SINH_HOC_12_TOC_MAPPING,
}

TOC_MAPPING = SINH_HOC_12_TOC_MAPPING


def get_toc_mapping(book_id):
    return BOOK_TOC_MAPPINGS.get(book_id, TOC_MAPPING)
