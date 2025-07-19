from pypdf import PdfReader
import json
import os

def extract_text_by_page(pdf_path):
    reader = PdfReader(pdf_path)
    all_data = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text:
            all_data.append((i + 1, text.splitlines()))
    return all_data

def analyze_headings(pages):
    # Simple heuristic: headings are short lines, possibly all caps or title case
    outline = []
    title = ""
    for page_num, lines in pages:
        for line in lines:
            line = line.strip()
            if not line or len(line) > 100:
                continue

            if page_num == 1 and not title and len(line.split()) <= 10:
                title = line

            if line.isupper() and len(line.split()) <= 6:
                level = "H1"
            elif line.istitle() and len(line.split()) <= 8:
                level = "H2"
            elif len(line.split()) <= 6:
                level = "H3"
            else:
                continue

            outline.append({
                "level": level,
                "text": line,
                "page": page_num
            })

    return {
        "title": title if title else "Untitled Document",
        "outline": outline
    }

def main():
    input_path = "C:\\Users\\Dell\\OneDrive\\Documents\\adobe1A\\InputPDF\\E0CCG5S239.pdf"  # PDF path
    output_path = "output/sample.json"

    pages = extract_text_by_page(input_path)
    structured = analyze_headings(pages)

    os.makedirs("output", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(structured, f, indent=2, ensure_ascii=False)

    print(" JSON saved to:", output_path)

if __name__ == "__main__":
    main()