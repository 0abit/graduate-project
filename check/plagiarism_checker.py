"""
╔══════════════════════════════════════════════════════════════╗
║        PLAGIARISM & AI DETECTION TOOL FOR DOCX FILES        ║
║  - Web Plagiarism Check (DuckDuckGo, không cần API Key)     ║
║  - AI Writing Detection (phân tích văn phong)               ║
╚══════════════════════════════════════════════════════════════╝
Cách dùng:
    python plagiarism_checker.py "path/to/file.docx"
    python plagiarism_checker.py  (sẽ hỏi nhập đường dẫn)
"""

import sys
import os
import re
import time
import math
import json
import random
import io
from datetime import datetime
from collections import Counter
from pathlib import Path

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from docx import Document
from duckduckgo_search import DDGS

# ════════════════════════════════════════════════════════════
#  CẤU HÌNH
# ════════════════════════════════════════════════════════════

# Số câu tối đa sẽ kiểm tra trên web (DuckDuckGo free, nhưng nên giới hạn)
MAX_WEB_CHECK_SENTENCES = 60

# Độ dài tối thiểu của câu để kiểm tra (câu quá ngắn thì không có giá trị)
MIN_SENTENCE_LENGTH = 30  # ký tự

# Thời gian chờ giữa mỗi lần search (tránh bị rate-limit)
SEARCH_DELAY_SECONDS = 2.5

# Ngưỡng phần trăm từ trùng khớp để coi là "đạo văn"
PLAGIARISM_WORD_OVERLAP_THRESHOLD = 0.6  # 60%

# Danh sách các cụm từ AI hay dùng (tiếng Việt)
AI_CLICHE_PHRASES = [
    "đóng vai trò quan trọng",
    "tuy nhiên",
    "mặt khác",
    "nhằm mục đích",
    "tối ưu hóa",
    "một cách hiệu quả",
    "có thể thấy rằng",
    "trong bối cảnh",
    "đặc biệt là",
    "cần phải lưu ý",
    "điều này cho thấy",
    "nói cách khác",
    "trên thực tế",
    "xét về mặt",
    "theo đó",
    "do đó",
    "vì vậy",
    "bên cạnh đó",
    "ngoài ra",
    "không chỉ",
    "mà còn",
    "đồng thời",
    "cụ thể là",
    "nói chung",
    "qua đó",
    "từ đó",
    "nhờ đó",
    "hơn nữa",
    "thêm vào đó",
    "chính vì vậy",
    "như đã đề cập",
    "theo như",
    "với mục đích",
    "góp phần",
    "nhìn chung",
    "xét cho cùng",
    "điều đáng chú ý",
    "rõ ràng rằng",
    "không thể phủ nhận",
    "cần được nhấn mạnh",
]

# Danh sách từ nối AI hay lạm dụng
AI_CONNECTORS = [
    "tuy nhiên", "mặt khác", "bên cạnh đó", "ngoài ra",
    "hơn nữa", "thêm vào đó", "đồng thời", "do đó",
    "vì vậy", "chính vì vậy", "nhờ đó", "qua đó", "từ đó",
    "theo đó", "cụ thể là",
]


# ════════════════════════════════════════════════════════════
#  1. ĐỌC FILE DOCX
# ════════════════════════════════════════════════════════════

def read_docx(filepath: str) -> list[str]:
    """Đọc file .docx và trả về danh sách các đoạn văn (paragraph)."""
    doc = Document(filepath)
    paragraphs = []
    for para in doc.paragraphs:
        text = para.text.strip()
        if text and len(text) > 15:  # Bỏ qua đoạn quá ngắn
            paragraphs.append(text)
    return paragraphs


def read_pdf(filepath: str) -> list[str]:
    """Đọc file .pdf và trả về danh sách các đoạn văn (paragraph)."""
    from pypdf import PdfReader
    paragraphs = []
    try:
        reader = PdfReader(filepath)
        for page in reader.pages:
            text = page.extract_text()
            if text:
                for para in text.split('\n'):
                    para = para.strip()
                    if len(para) > 15:
                        paragraphs.append(para)
    except Exception as e:
        print(f"Lỗi đọc PDF: {e}")
    return paragraphs


def split_sentences(text: str) -> list[str]:
    """Tách đoạn văn thành các câu riêng lẻ."""
    # Tách theo dấu chấm, chấm hỏi, chấm than, chấm phẩy
    # Giữ lại những câu đủ dài
    raw_sentences = re.split(r'[.!?]\s+', text)
    sentences = []
    for s in raw_sentences:
        s = s.strip()
        if len(s) >= MIN_SENTENCE_LENGTH:
            sentences.append(s)
    return sentences


# ════════════════════════════════════════════════════════════
#  2. KIỂM TRA ĐẠO VĂN TRÊN WEB (DuckDuckGo)
# ════════════════════════════════════════════════════════════

def normalize_text(text: str) -> str:
    """Chuẩn hóa text: lowercase, bỏ dấu câu thừa."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\sàáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđ]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def word_overlap_ratio(text1: str, text2: str) -> float:
    """Tính tỷ lệ từ trùng lặp giữa 2 đoạn text."""
    words1 = set(normalize_text(text1).split())
    words2 = set(normalize_text(text2).split())
    if not words1:
        return 0.0
    overlap = words1 & words2
    return len(overlap) / len(words1)


def search_web_for_sentence(sentence: str, ddgs: DDGS) -> list[dict]:
    """Tìm kiếm một câu trên DuckDuckGo và trả về kết quả."""
    try:
        # Đặt câu trong dấu ngoặc kép để tìm chính xác
        # Nếu câu quá dài, lấy 12 từ đầu tiên
        words = sentence.split()
        if len(words) > 12:
            search_query = ' '.join(words[:12])
        else:
            search_query = sentence

        results = ddgs.text(
            f'"{search_query}"',
            max_results=5,
            region='vn-vi'
        )
        return results if results else []
    except Exception as e:
        # Nếu bị rate-limit hoặc lỗi, chờ lâu hơn rồi thử lại
        if "ratelimit" in str(e).lower() or "429" in str(e):
            print(f"    ⏳ DuckDuckGo rate-limit, chờ 10s rồi thử lại...")
            time.sleep(10)
            try:
                results = ddgs.text(f'"{search_query}"', max_results=3, region='vn-vi')
                return results if results else []
            except:
                return []
        return []


def check_web_plagiarism(sentences: list[str]) -> list[dict]:
    """
    Kiểm tra đạo văn trên web cho danh sách câu.
    Trả về danh sách các câu nghi ngờ đạo văn.
    """
    print("\n" + "=" * 60)
    print("🌐 KIỂM TRA ĐẠO VĂN TRÊN WEB (DuckDuckGo)")
    print("=" * 60)

    # Chọn ngẫu nhiên nếu quá nhiều câu
    if len(sentences) > MAX_WEB_CHECK_SENTENCES:
        print(f"📊 Tổng {len(sentences)} câu, chọn ngẫu nhiên {MAX_WEB_CHECK_SENTENCES} câu để kiểm tra.")
        selected = random.sample(sentences, MAX_WEB_CHECK_SENTENCES)
    else:
        selected = sentences
        print(f"📊 Kiểm tra {len(selected)} câu trên web.")

    plagiarism_results = []

    with DDGS() as ddgs:
        for i, sentence in enumerate(selected):
            progress = f"[{i+1}/{len(selected)}]"
            short_display = sentence[:60] + "..." if len(sentence) > 60 else sentence
            print(f"\n  {progress} 🔍 \"{short_display}\"")

            results = search_web_for_sentence(sentence, ddgs)

            if results:
                for r in results:
                    body = r.get('body', '')
                    title = r.get('title', '')
                    href = r.get('href', '')
                    overlap = word_overlap_ratio(sentence, body)

                    if overlap >= PLAGIARISM_WORD_OVERLAP_THRESHOLD:
                        plagiarism_results.append({
                            'sentence': sentence,
                            'source_title': title,
                            'source_url': href,
                            'source_snippet': body[:200],
                            'overlap_ratio': overlap,
                        })
                        print(f"    ⚠️  TRÙNG {overlap:.0%} — {href[:80]}")
                        break  # Chỉ cần 1 nguồn trùng là đủ
                else:
                    print(f"    ✅ Không tìm thấy trùng lặp đáng kể.")
            else:
                print(f"    ✅ Không có kết quả trên web.")

            # Delay để tránh bị rate-limit
            time.sleep(SEARCH_DELAY_SECONDS)

    return plagiarism_results


# ════════════════════════════════════════════════════════════
#  3. PHÁT HIỆN VĂN PHONG AI
# ════════════════════════════════════════════════════════════

def count_cliche_phrases(text: str) -> list[dict]:
    """Đếm số lần xuất hiện của các cụm từ rập khuôn AI."""
    text_lower = text.lower()
    found = []
    for phrase in AI_CLICHE_PHRASES:
        count = text_lower.count(phrase.lower())
        if count > 0:
            found.append({'phrase': phrase, 'count': count})
    found.sort(key=lambda x: x['count'], reverse=True)
    return found


def calculate_burstiness(sentences: list[str]) -> dict:
    """
    Đo Burstiness — sự xen kẽ ngẫu nhiên giữa câu ngắn và dài.
    AI thường viết câu có độ dài đều đều nhau (burstiness thấp).
    Con người viết câu lúc ngắn lúc dài hơn (burstiness cao).
    """
    if len(sentences) < 3:
        return {'score': 0, 'verdict': 'Không đủ dữ liệu', 'details': {}}

    lengths = [len(s.split()) for s in sentences]
    mean_len = sum(lengths) / len(lengths)
    variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
    std_dev = math.sqrt(variance)

    # Coefficient of Variation (CV) = std / mean
    cv = std_dev / mean_len if mean_len > 0 else 0

    # Đếm câu ngắn (<10 từ) và câu dài (>25 từ)
    short_count = sum(1 for l in lengths if l < 10)
    long_count = sum(1 for l in lengths if l > 25)

    # Score: CV cao = burstiness cao = giống người viết
    if cv > 0.6:
        verdict = "🟢 Burstiness CAO — giống văn phong con người"
    elif cv > 0.35:
        verdict = "🟡 Burstiness TRUNG BÌNH — có thể là người hoặc AI đã chỉnh sửa"
    else:
        verdict = "🔴 Burstiness THẤP — rất giống AI (câu đều đều nhau)"

    return {
        'score': round(cv, 3),
        'verdict': verdict,
        'details': {
            'mean_words_per_sentence': round(mean_len, 1),
            'std_dev': round(std_dev, 1),
            'total_sentences': len(sentences),
            'short_sentences_under_10_words': short_count,
            'long_sentences_over_25_words': long_count,
            'min_length': min(lengths),
            'max_length': max(lengths),
        }
    }


def calculate_type_token_ratio(text: str) -> dict:
    """
    Tính Type-Token Ratio (TTR) — tỷ lệ từ vựng đa dạng.
    AI thường có TTR thấp hơn vì hay lặp lại từ vựng.
    """
    words = normalize_text(text).split()
    if not words:
        return {'score': 0, 'verdict': 'Không đủ dữ liệu'}

    total_words = len(words)
    unique_words = len(set(words))
    ttr = unique_words / total_words

    if ttr > 0.55:
        verdict = "🟢 TTR CAO — từ vựng đa dạng, giống con người"
    elif ttr > 0.40:
        verdict = "🟡 TTR TRUNG BÌNH"
    else:
        verdict = "🔴 TTR THẤP — từ vựng lặp nhiều, có dấu hiệu AI"

    return {
        'score': round(ttr, 3),
        'verdict': verdict,
        'details': {
            'total_words': total_words,
            'unique_words': unique_words,
        }
    }


def count_connector_density(text: str) -> dict:
    """
    Đếm mật độ từ nối — AI lạm dụng từ nối rất nhiều.
    """
    text_lower = text.lower()
    total_words = len(text_lower.split())
    connector_count = 0

    for connector in AI_CONNECTORS:
        connector_count += text_lower.count(connector.lower())

    density = connector_count / total_words * 100 if total_words > 0 else 0

    if density < 1.5:
        verdict = "🟢 Mật độ từ nối THẤP — giống con người"
    elif density < 3.0:
        verdict = "🟡 Mật độ từ nối TRUNG BÌNH"
    else:
        verdict = "🔴 Mật độ từ nối CAO — AI hay lạm dụng từ nối"

    return {
        'score': round(density, 2),
        'verdict': verdict,
        'details': {
            'connector_count': connector_count,
            'total_words': total_words,
            'density_percent': f"{density:.2f}%",
        }
    }


def calculate_sentence_start_diversity(sentences: list[str]) -> dict:
    """
    Đo mức độ đa dạng của từ mở đầu câu.
    AI hay bắt đầu câu bằng cùng một cách.
    """
    if len(sentences) < 5:
        return {'score': 0, 'verdict': 'Không đủ dữ liệu'}

    starters = []
    for s in sentences:
        words = s.strip().split()
        if words:
            # Lấy 2 từ đầu tiên
            starter = ' '.join(words[:min(2, len(words))]).lower()
            starters.append(starter)

    total = len(starters)
    unique = len(set(starters))
    diversity = unique / total if total > 0 else 0

    # Đếm các từ mở đầu phổ biến nhất
    counter = Counter(starters)
    top_5 = counter.most_common(5)

    if diversity > 0.7:
        verdict = "🟢 Cách mở đầu câu ĐA DẠNG — giống con người"
    elif diversity > 0.5:
        verdict = "🟡 Cách mở đầu câu TRUNG BÌNH"
    else:
        verdict = "🔴 Cách mở đầu câu LẶP LẠI — dấu hiệu AI"

    return {
        'score': round(diversity, 3),
        'verdict': verdict,
        'details': {
            'total_sentences': total,
            'unique_starters': unique,
            'top_5_starters': [{'starter': s, 'count': c} for s, c in top_5],
        }
    }


def ai_detection_analysis(paragraphs: list[str]) -> dict:
    """Chạy toàn bộ phân tích AI detection."""
    print("\n" + "=" * 60)
    print("🤖 PHÂN TÍCH VĂN PHONG AI")
    print("=" * 60)

    full_text = ' '.join(paragraphs)
    all_sentences = []
    for p in paragraphs:
        all_sentences.extend(split_sentences(p))

    # 1. Burstiness
    print("\n📐 1. Burstiness (xen kẽ câu ngắn/dài):")
    burstiness = calculate_burstiness(all_sentences)
    print(f"   Score (CV): {burstiness['score']}")
    print(f"   {burstiness['verdict']}")
    d = burstiness['details']
    if d:
        print(f"   Trung bình: {d['mean_words_per_sentence']} từ/câu | "
              f"Ngắn nhất: {d['min_length']} từ | Dài nhất: {d['max_length']} từ")
        print(f"   Câu ngắn (<10 từ): {d['short_sentences_under_10_words']} | "
              f"Câu dài (>25 từ): {d['long_sentences_over_25_words']}")

    # 2. Type-Token Ratio
    print("\n📊 2. Type-Token Ratio (đa dạng từ vựng):")
    ttr = calculate_type_token_ratio(full_text)
    print(f"   Score: {ttr['score']}")
    print(f"   {ttr['verdict']}")
    if 'details' in ttr:
        print(f"   Tổng từ: {ttr['details']['total_words']} | "
              f"Từ duy nhất: {ttr['details']['unique_words']}")

    # 3. Mật độ từ nối
    print("\n🔗 3. Mật độ từ nối (connector density):")
    connectors = count_connector_density(full_text)
    print(f"   Score: {connectors['score']}%")
    print(f"   {connectors['verdict']}")

    # 4. Đa dạng từ mở đầu câu
    print("\n✏️  4. Đa dạng cách mở đầu câu:")
    starters = calculate_sentence_start_diversity(all_sentences)
    print(f"   Score: {starters['score']}")
    print(f"   {starters['verdict']}")
    if 'details' in starters and 'top_5_starters' in starters['details']:
        print(f"   Top 5 cách mở đầu phổ biến nhất:")
        for item in starters['details']['top_5_starters']:
            print(f"      \"{item['starter']}...\" — {item['count']} lần")

    # 5. Cụm từ rập khuôn
    print("\n📝 5. Cụm từ rập khuôn AI:")
    cliches = count_cliche_phrases(full_text)
    total_cliches = sum(c['count'] for c in cliches)
    word_count = len(full_text.split())
    cliche_density = total_cliches / word_count * 100 if word_count > 0 else 0

    if cliche_density < 1.0:
        cliche_verdict = "🟢 Ít cụm từ rập khuôn — giống con người"
    elif cliche_density < 2.5:
        cliche_verdict = "🟡 Mức trung bình"
    else:
        cliche_verdict = "🔴 Nhiều cụm từ rập khuôn — dấu hiệu AI"

    print(f"   Tổng: {total_cliches} cụm từ rập khuôn / {word_count} từ ({cliche_density:.2f}%)")
    print(f"   {cliche_verdict}")
    if cliches[:10]:
        print(f"   Top 10 cụm từ phổ biến nhất:")
        for c in cliches[:10]:
            print(f"      \"{c['phrase']}\" — {c['count']} lần")

    # ═══ TỔNG KẾT AI SCORE ═══
    print("\n" + "─" * 60)
    print("📋 TỔNG KẾT AI DETECTION")
    print("─" * 60)

    # Tính điểm tổng hợp (0-100, 100 = chắc chắn AI)
    ai_score = 0

    # Burstiness: CV thấp = AI
    if burstiness['score'] < 0.35:
        ai_score += 30
    elif burstiness['score'] < 0.6:
        ai_score += 15

    # TTR thấp = AI
    if ttr['score'] < 0.40:
        ai_score += 20
    elif ttr['score'] < 0.55:
        ai_score += 10

    # Connector density cao = AI
    if connectors['score'] > 3.0:
        ai_score += 20
    elif connectors['score'] > 1.5:
        ai_score += 10

    # Starter diversity thấp = AI
    if starters['score'] < 0.5:
        ai_score += 15
    elif starters['score'] < 0.7:
        ai_score += 7

    # Cliche density cao = AI
    if cliche_density > 2.5:
        ai_score += 15
    elif cliche_density > 1.0:
        ai_score += 7

    if ai_score >= 60:
        overall = "🔴 XÁC SUẤT CAO LÀ AI VIẾT"
    elif ai_score >= 35:
        overall = "🟡 CÓ DẤU HIỆU AI — cần chỉnh sửa thêm"
    else:
        overall = "🟢 VĂN PHONG TỰ NHIÊN — ít dấu hiệu AI"

    print(f"\n   🎯 AI SCORE: {ai_score}/100")
    print(f"   {overall}")

    return {
        'ai_score': ai_score,
        'overall_verdict': overall,
        'burstiness': burstiness,
        'ttr': ttr,
        'connector_density': connectors,
        'starter_diversity': starters,
        'cliche_phrases': cliches,
        'cliche_density_percent': round(cliche_density, 2),
    }


# ════════════════════════════════════════════════════════════
#  4. BÁO CÁO KẾT QUẢ
# ════════════════════════════════════════════════════════════

def generate_report(filepath: str, plagiarism_results: list[dict],
                    ai_results: dict, total_sentences: int):
    """Tạo file báo cáo kết quả."""
    report_path = os.path.join(os.path.dirname(filepath),
                               f"_plagiarism_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt")

    with open(report_path, 'w', encoding='utf-8') as f:
        f.write("=" * 70 + "\n")
        f.write("       BÁO CÁO KIỂM TRA ĐẠO VĂN & AI DETECTION\n")
        f.write(f"       File: {os.path.basename(filepath)}\n")
        f.write(f"       Thời gian: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        # ─── WEB PLAGIARISM ───
        f.write("╔" + "═" * 50 + "╗\n")
        f.write("║  1. KẾT QUẢ KIỂM TRA ĐẠO VĂN WEB\n")
        f.write("╚" + "═" * 50 + "╝\n\n")

        plag_count = len(plagiarism_results)
        plag_rate = plag_count / total_sentences * 100 if total_sentences > 0 else 0

        f.write(f"  Tổng câu kiểm tra: {min(total_sentences, MAX_WEB_CHECK_SENTENCES)}\n")
        f.write(f"  Câu nghi ngờ đạo văn: {plag_count}\n")
        f.write(f"  Tỷ lệ: {plag_rate:.1f}%\n\n")

        if plagiarism_results:
            for i, pr in enumerate(plagiarism_results, 1):
                f.write(f"  [{i}] TRÙNG {pr['overlap_ratio']:.0%}\n")
                f.write(f"      Câu gốc: \"{pr['sentence'][:120]}...\"\n")
                f.write(f"      Nguồn:   {pr['source_title'][:80]}\n")
                f.write(f"      URL:     {pr['source_url']}\n")
                f.write(f"      Snippet: \"{pr['source_snippet'][:150]}...\"\n\n")
        else:
            f.write("  ✅ Không phát hiện đạo văn từ web.\n\n")

        # ─── AI DETECTION ───
        f.write("\n╔" + "═" * 50 + "╗\n")
        f.write("║  2. KẾT QUẢ PHÂN TÍCH VĂN PHONG AI\n")
        f.write("╚" + "═" * 50 + "╝\n\n")

        f.write(f"  🎯 AI SCORE: {ai_results['ai_score']}/100\n")
        f.write(f"  {ai_results['overall_verdict']}\n\n")

        f.write(f"  Burstiness (CV):      {ai_results['burstiness']['score']} — "
                f"{ai_results['burstiness']['verdict']}\n")
        f.write(f"  Type-Token Ratio:     {ai_results['ttr']['score']} — "
                f"{ai_results['ttr']['verdict']}\n")
        f.write(f"  Connector Density:    {ai_results['connector_density']['score']}% — "
                f"{ai_results['connector_density']['verdict']}\n")
        f.write(f"  Starter Diversity:    {ai_results['starter_diversity']['score']} — "
                f"{ai_results['starter_diversity']['verdict']}\n")
        f.write(f"  Cliché Density:       {ai_results['cliche_density_percent']}%\n\n")

        if ai_results['cliche_phrases']:
            f.write("  Cụm từ rập khuôn tìm thấy:\n")
            for c in ai_results['cliche_phrases'][:15]:
                f.write(f"    - \"{c['phrase']}\" × {c['count']} lần\n")

        f.write("\n" + "=" * 70 + "\n")
        f.write("  LƯU Ý: Đây là công cụ hỗ trợ, KHÔNG thay thế kiểm tra chuyên nghiệp.\n")
        f.write("  Các kết quả chỉ mang tính tham khảo.\n")
        f.write("=" * 70 + "\n")

    print(f"\n📄 Báo cáo đã lưu tại: {report_path}")
    return report_path


# ════════════════════════════════════════════════════════════
#  5. MAIN
# ════════════════════════════════════════════════════════════

def main():
    print("╔══════════════════════════════════════════════════════╗")
    print("║    🔍 PLAGIARISM & AI DETECTION TOOL v1.0           ║")
    print("║    Kiểm tra đạo văn + Phát hiện văn phong AI       ║")
    print("╚══════════════════════════════════════════════════════╝")

    # Lấy đường dẫn file
    if len(sys.argv) > 1:
        filepath = sys.argv[1]
    else:
        filepath = input("\n📂 Nhập đường dẫn file .docx: ").strip().strip('"')

    if not os.path.exists(filepath):
        print(f"❌ Không tìm thấy file: {filepath}")
        sys.exit(1)

    if not filepath.lower().endswith('.docx') and not filepath.lower().endswith('.pdf'):
        print(f"❌ File phải có đuôi .docx hoặc .pdf")
        sys.exit(1)

    # Đọc file
    print(f"\n📖 Đang đọc file: {os.path.basename(filepath)}...")
    if filepath.lower().endswith('.docx'):
        paragraphs = read_docx(filepath)
    else:
        paragraphs = read_pdf(filepath)
    print(f"   Tìm thấy {len(paragraphs)} đoạn văn.")

    # Tách câu
    all_sentences = []
    for p in paragraphs:
        all_sentences.extend(split_sentences(p))
    print(f"   Tách được {len(all_sentences)} câu (>= {MIN_SENTENCE_LENGTH} ký tự).")

    # Hỏi muốn chạy phần nào
    print("\n🔧 Chọn chế độ kiểm tra:")
    print("   [1] Chỉ kiểm tra đạo văn web")
    print("   [2] Chỉ phân tích AI detection")
    print("   [3] Cả hai (đầy đủ)")
    choice = input("\n   Chọn (1/2/3, mặc định 3): ").strip() or "3"

    plagiarism_results = []
    ai_results = None

    if choice in ('1', '3'):
        plagiarism_results = check_web_plagiarism(all_sentences)

    if choice in ('2', '3'):
        ai_results = ai_detection_analysis(paragraphs)

    # Tạo báo cáo
    if ai_results is None:
        ai_results = {
            'ai_score': -1,
            'overall_verdict': 'Không chạy',
            'burstiness': {'score': 0, 'verdict': 'N/A'},
            'ttr': {'score': 0, 'verdict': 'N/A'},
            'connector_density': {'score': 0, 'verdict': 'N/A'},
            'starter_diversity': {'score': 0, 'verdict': 'N/A'},
            'cliche_phrases': [],
            'cliche_density_percent': 0,
        }

    report_path = generate_report(filepath, plagiarism_results,
                                  ai_results, len(all_sentences))

    # Tổng kết
    print("\n" + "═" * 60)
    print("🏁 HOÀN TẤT!")
    print("═" * 60)
    if plagiarism_results:
        print(f"   ⚠️  Tìm thấy {len(plagiarism_results)} câu nghi ngờ đạo văn")
    else:
        print(f"   ✅ Không phát hiện đạo văn từ web")
    if ai_results and ai_results['ai_score'] >= 0:
        print(f"   🤖 AI Score: {ai_results['ai_score']}/100")
    print(f"   📄 Báo cáo: {report_path}")
    print("═" * 60)


if __name__ == '__main__':
    main()
