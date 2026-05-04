import fitz  # PyMuPDF
import easyocr
import pandas as pd
import re
import os
from difflib import get_close_matches
 
PDF_FILES = [
    "doctor_receipts_1.pdf",
    "doctor_receipts_2.pdf",
    "doctor_receipts_3.pdf",
    "doctor_receipts_4.pdf",
    "doctor_receipts_5.pdf"
]
 
MODEL_PATH = r"C:\Users\Jalen\.EasyOCR\model"
OUTPUT_FILE = "Invoice_Extract.xlsx"
CONFIDENCE_THRESHOLD = 0.4   # drop OCR results below this score
 
reader = easyocr.Reader(
    ['en'],
    gpu=False,
    model_storage_directory=MODEL_PATH,
    download_enabled=False
)
 
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []
    for page_num, page in enumerate(doc, start=1):
        mat = fitz.Matrix(2, 2)
        pix = page.get_pixmap(matrix=mat)
        img_path = f"temp_page_{page_num}.png"
        pix.save(img_path)
        images.append((page_num, img_path))
    return images
 
def get_bbox_center(bbox):
    """Return (cx, cy) center of an EasyOCR bounding box."""
    xs = [p[0] for p in bbox]
    ys = [p[1] for p in bbox]
    return (sum(xs) / 4, sum(ys) / 4)
 
def sort_by_position(results):
    """Sort OCR results top-to-bottom, then left-to-right."""
    return sorted(results, key=lambda r: (round(get_bbox_center(r[0])[1] / 15), get_bbox_center(r[0])[0]))
 
def group_into_lines(results, y_tolerance=15):
    """
    Group OCR tokens into lines based on vertical proximity.
    Returns list of lines, each line is a list of (bbox, text, conf) sorted left-to-right.
    """
    if not results:
        return []
 
    sorted_results = sort_by_position(results)
    lines = []
    current_line = [sorted_results[0]]
    current_y = get_bbox_center(sorted_results[0][0])[1]
 
    for item in sorted_results[1:]:
        item_y = get_bbox_center(item[0])[1]
        if abs(item_y - current_y) <= y_tolerance:
            current_line.append(item)
        else:
            lines.append(sorted(current_line, key=lambda r: get_bbox_center(r[0])[0]))
            current_line = [item]
            current_y = item_y
 
    lines.append(sorted(current_line, key=lambda r: get_bbox_center(r[0])[0]))
    return lines
 
def line_text(line):
    """Join tokens in a line into a string."""
    return " ".join(item[1] for item in line)
 
def normalize(text):
    """Fix common OCR substitutions in Philippine receipts."""
    fixes = {
        r"\b0(?=[A-Z])": "O",   
        r"(?<=[A-Z])0\b": "O",  
        r"\bl(?=\d)": "1",      
        r"(?<=\d)l\b": "1",
        r"\bS(?=\d)": "5",
        r"\|": "I",
    }
    for pattern, replacement in fixes.items():
        text = re.sub(pattern, replacement, text)
    return text
 
LABEL_KEYWORDS = {
    "receipt_no":   ["receipt no", "receipt #", "or no", "or #", "official receipt", "invoice no"],
    "doctor_name":  ["dr.", "dr ", "physician", "attending", "doctor"],
    "prc_license":  ["prc", "ptr", "license no", "lic. no", "prc no"],
    "hospital":     ["hospital", "clinic", "medical center", "health center", "infirmary"],
    "date":         ["date", "issued", "date issued"],
    "patient_name": ["patient", "patient name", "pt.", "pt name", "name of patient"],
    "total_amount": ["total", "amount due", "amount paid", "grand total", "total amount"],
}
 
def fuzzy_match_label(text, keyword_list, threshold=0.6):
    """Check if text fuzzy-matches any label keyword."""
    text_lower = text.lower().strip()
    for kw in keyword_list:
        if kw in text_lower:
            return True
        words = text_lower.split()
        kw_words = kw.split()
        if any(get_close_matches(w, kw_words, n=1, cutoff=threshold) for w in words):
            return True
    return False
 
def extract_value_after_label(line_text_str, label_keywords):
    """Extract the value part from a line like 'Patient Name: Juan dela Cruz'."""
    if ":" in line_text_str:
        parts = line_text_str.split(":", 1)
        if fuzzy_match_label(parts[0], label_keywords):
            return parts[1].strip()
    if " - " in line_text_str:
        parts = line_text_str.split(" - ", 1)
        if fuzzy_match_label(parts[0], label_keywords):
            return parts[1].strip()
    return ""
 
def extract_receipt_no(text):
    patterns = [
        r"(?:OR|O\.R\.|Receipt|Invoice)\s*(?:No\.?|#|Na\.?)\s*[:.\-]?\s*([A-Z0-9\-]+)",
        r"(?:No|Na|Num)\s*[:.\-]\s*([A-Z]{1,4}\d{4,})",
        r"\b(OR\d{4,})\b",
        r"\b([A-Z]{1,4}[\s\-]?\d{6,})\b",
        r"\b(\d{6,})\b",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().replace(" ", "")
    return ""
 
def extract_date(text):
    patterns = [
        r"(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})",                        
        r"(\d{1,2}[\/\-\.]?\d{1,2}[\/\-\.]?\d{4})",                        
        r"(\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})",   
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{1,2},?\s*\d{4})", 
        r"((?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\.?\s+\d{4})",             
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip()
    return ""
 
def extract_prc(text):
    patterns = [
        r"(?:PRC|PTR|LIC|License)\s*(?:Lic\.?|No\.?|#)?\s*[:.\-]?\s*([A-Z]{0,3}\s*\d{4,})",
        r"(?:PRC|PTR)\b.*?([A-Z]{0,3}\s*\d{4,})",
    ]
    for p in patterns:
        m = re.search(p, text, re.IGNORECASE)
        if m:
            return m.group(1).strip().replace(" ", "")
    return ""
 
def normalize_amount(raw):
    raw = raw.strip()
    if re.match(r"^\d{1,4},\d{2}$", raw):
        return float(raw.replace(",", "."))
    if re.match(r"^\d{1,3}(?:\.\d{3})+,\d{2}$", raw):
        return float(raw.replace(".", "").replace(",", "."))
    if re.match(r"^\d{1,3}(?:,\d{3})+\.\d{2}$", raw):
        return float(raw.replace(",", ""))
    if re.match(r"^\d+\.\d{2}$", raw):
        return float(raw)
    return None
 
AMOUNT_RE = re.compile(
    r"\b(\d{1,3}(?:[.,]\d{3})*[.,]\d{2}|\d{1,4}[.,]\d{2})\b"
)
 
def extract_amount(text):
    lines = text.split("\n")
 
    for i, line in enumerate(lines):
        if fuzzy_match_label(line, LABEL_KEYWORDS["total_amount"]):
            candidates = AMOUNT_RE.findall(line)
            if candidates:
                amounts = [normalize_amount(c) for c in candidates]
                amounts = [a for a in amounts if a is not None and a > 0]
                if amounts:
                    return f"{max(amounts):,.2f}"
            if i + 1 < len(lines):
                candidates = AMOUNT_RE.findall(lines[i + 1])
                if candidates:
                    amounts = [normalize_amount(c) for c in candidates]
                    amounts = [a for a in amounts if a is not None and a > 0]
                    if amounts:
                        return f"{max(amounts):,.2f}"
 
    php_amounts = re.findall(
        r"PHP\s*(\d{1,3}(?:[.,]\d{3})*[.,]\d{2}|\d{1,4}[.,]\d{2})",
        text, re.IGNORECASE
    )
    if php_amounts:
        normalized = [normalize_amount(a) for a in php_amounts]
        normalized = [a for a in normalized if a is not None and a > 0]
        if normalized:
            best = max(normalized)
            if best < 1_000_000:
                return f"{best:,.2f}"
 
    candidates = AMOUNT_RE.findall(text)
    amounts = [(normalize_amount(c), c) for c in candidates]
    amounts = [(v, c) for v, c in amounts if v is not None and 0 < v < 1_000_000]
    if amounts:
        return f"{max(amounts, key=lambda x: x[0])[0]:,.2f}"
    return ""
 
def extract_name_from_line(line_str, skip_keywords=None):
    """Extract a proper name (2+ capitalized words) from a line."""
    skip_keywords = skip_keywords or []
    for kw in skip_keywords:
        line_str = re.sub(re.escape(kw), "", line_str, flags=re.IGNORECASE)
    line_str = re.sub(r"[:\-,]", " ", line_str).strip()
    m = re.search(r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,4})", line_str)
    return m.group(1).strip() if m else line_str.strip()
 
def extract_fields(page_num, ocr_results):
    filtered = [(bbox, text, conf) for bbox, text, conf in ocr_results if conf >= CONFIDENCE_THRESHOLD]
 
    lines = group_into_lines(filtered)
 
    full_text = "\n".join(normalize(line_text(line)) for line in lines)
 
    fields = {
        "Page":              page_num,
        "Receipt No.":       "",
        "Doctor Name":       "",
        "PRC License":       "",
        "Hospital":          "",
        "Date":              "",
        "Patient Name":      "",
        "Total Amount (PHP)": "",
    }
 
    seen_lines = set()
 
    for i, line in enumerate(lines):
        lt = normalize(line_text(line))
        if lt in seen_lines:
            continue
        seen_lines.add(lt)
 
        next_lt = normalize(line_text(lines[i + 1])) if i + 1 < len(lines) else ""
 
        if not fields["Receipt No."]:
            val = extract_value_after_label(lt, LABEL_KEYWORDS["receipt_no"])
            if val:
                fields["Receipt No."] = val
            elif fuzzy_match_label(lt, LABEL_KEYWORDS["receipt_no"]) and next_lt:
                m = re.search(r"[A-Z0-9\-]{4,}", next_lt)
                if m:
                    fields["Receipt No."] = m.group(0)
            else:
                rn = extract_receipt_no(lt)
                if rn:
                    fields["Receipt No."] = rn
 
        if not fields["Doctor Name"]:
            if fuzzy_match_label(lt, LABEL_KEYWORDS["doctor_name"]):
                val = extract_value_after_label(lt, LABEL_KEYWORDS["doctor_name"])
                if val:
                    fields["Doctor Name"] = val
                else:
                    m = re.search(r"Dr\.?\s+([A-Z][a-zA-Z\s]+)", lt, re.IGNORECASE)
                    if m:
                        fields["Doctor Name"] = "Dr. " + m.group(1).strip()
 
        if not fields["PRC License"]:
            prc = extract_prc(lt)
            if prc:
                fields["PRC License"] = prc
 
        if not fields["Hospital"]:
            if fuzzy_match_label(lt, LABEL_KEYWORDS["hospital"]):
                fields["Hospital"] = lt.strip()
 
        if not fields["Date"]:
            val = extract_value_after_label(lt, LABEL_KEYWORDS["date"])
            if val:
                d = extract_date(val)
                if d:
                    fields["Date"] = d
            if not fields["Date"]:
                d = extract_date(lt)
                if d:
                    fields["Date"] = d
 
        if not fields["Patient Name"]:
            if fuzzy_match_label(lt, LABEL_KEYWORDS["patient_name"]):
                val = extract_value_after_label(lt, LABEL_KEYWORDS["patient_name"])
                if val:
                    fields["Patient Name"] = val
                elif next_lt:
                    fields["Patient Name"] = extract_name_from_line(next_lt)
 
    fields["Total Amount (PHP)"] = extract_amount(full_text)
 
    return fields
 
records = []
global_page = 1
 
for pdf in PDF_FILES:
    if not os.path.exists(pdf):
        print(f"⚠️  Skipping {pdf} (not found)")
        continue
 
    print(f"\nProcessing {pdf}...")
    pages = pdf_to_images(pdf)
 
    for _, img_path in pages:
        print(f"  OCR Page {global_page}...")
        ocr_results = reader.readtext(img_path)
        record = extract_fields(global_page, ocr_results)
        records.append(record)
        os.remove(img_path)
        global_page += 1
 
df = pd.DataFrame(records)
 
cols = ["Page", "Receipt No.", "Doctor Name", "PRC License", "Hospital", "Date", "Patient Name", "Total Amount (PHP)"]
df = df[[c for c in cols if c in df.columns]]
 
df.to_excel(OUTPUT_FILE, index=False)
print(f"\n✅ Done! Output saved to {OUTPUT_FILE}")