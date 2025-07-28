from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextContainer, LTChar, LTTextLine
import pandas as pd
import os
import re


def extract_pdf_features(pdf_path):
    rows = []
    for page_num, page_layout in enumerate(extract_pages(pdf_path), start=1):
        last_y1 = None  # for line spacing

        for element in page_layout:
            if isinstance(element, LTTextContainer):
                for text_line in element:
                    if isinstance(text_line, LTTextLine):
                        line_text = text_line.get_text().strip()
                        if not line_text:
                            continue
    
                        font_sizes = []
                        fonts = []
                        x0s, x1s, y0s, y1s = [], [], [], []

                        for char in text_line:
                            if isinstance(char, LTChar):
                                font_sizes.append(char.size)
                                fonts.append(char.fontname)
                                x0s.append(char.x0)
                                x1s.append(char.x1)
                                y0s.append(char.y0)
                                y1s.append(char.y1)

                        if not font_sizes:
                            continue

                        avg_font_size = sum(font_sizes) / len(font_sizes)
                        font_name = max(set(fonts), key=fonts.count) if fonts else ""
                        is_bold = "Bold" in font_name or "bold" in font_name
                        is_italic = "Italic" in font_name or "Oblique" in font_name

                        x0 = min(x0s) if x0s else None
                        x1 = max(x1s) if x1s else None
                        y0 = min(y0s) if y0s else None
                        y1 = max(y1s) if y1s else None
                        y_center = (y0 + y1) / 2 if y0 and y1 else None

                        # === Derived Features ===
                        alignment = "left"
                        if x0 > 200 and x1 < 400:
                            alignment = "center"
                        elif x0 > 400:
                            alignment = "right"

                        line_spacing = None
                        if last_y1 is not None and y1 is not None:
                            line_spacing = abs(last_y1 - y1)
                        last_y1 = y1

                        is_uppercase = line_text.isupper()
                        num_words = len(line_text.split())
                        text_length = len(line_text)
                        contains_colon_or_dot = (":" in line_text or "." in line_text)
                        is_numbered_heading = bool(re.match(r"^\d+(\.\d+)*\s", line_text))
                        has_bullets_or_dashes = bool(re.match(r"^\s*[\u2022\u2023\u25E6\-–\*]+\s", line_text))

                        rows.append({
                            "text": line_text,
                            "font_size": avg_font_size,
                            "font_name": font_name,
                            "is_bold": int(is_bold),
                            "is_italic": int(is_italic),
                            "x0": x0,
                            "x1": x1,
                            "y0": y0,
                            "y1": y1,
                            "y_pos": y_center,
                            "text_alignment": alignment,
                            "line_spacing": line_spacing,
                            "page": page_num,
                            "is_uppercase": int(is_uppercase),
                            "num_words": num_words,
                            "text_length": text_length,
                            "contains_colon_or_dot": int(contains_colon_or_dot),
                            "is_numbered_heading": int(is_numbered_heading),
                            "has_bullets_or_dashes": int(has_bullets_or_dashes),
                        })
    return rows


def process_pdfs(pdf_folder, output_csv):
    all_rows = []
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            pdf_path = os.path.join(pdf_folder, filename)
            print(f"Processing: {filename}")
            rows = extract_pdf_features(pdf_path)
            for row in rows:
                row["pdf_file"] = filename
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(output_csv, index=False)
    print(f"\n Done! Extracted features saved to: {output_csv}")



# === CONFIG ===
PDF_FOLDER = "app/INPUT"  # Folder containing your PDFs

OUTPUT_CSV = "combined_unlabeled.csv"  # Output path

if __name__ == "__main__":
    process_pdfs(PDF_FOLDER, OUTPUT_CSV)