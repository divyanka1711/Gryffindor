import fitz  # PyMuPDF
from joblib import load
import json

# Load your trained model
model = load("heading_model.joblib")

# Reverse label map
label_map_rev = {0: "H1", 1: "H2", 2: "H3", 3: "TEXT"}

def predict_from_pdf(pdf_path):
    doc = fitz.open(pdf_path)
    outline = []

    for page_num, page in enumerate(doc, start=1):
        blocks = page.get_text("dict")["blocks"]
        for block in blocks:
            if "lines" not in block:
                continue
            for line in block["lines"]:
                spans = line["spans"]
                if not spans:
                    continue

                text = " ".join([span["text"] for span in spans]).strip()
                if not text or len(text) > 100:
                    continue

                font_size = spans[0]["size"]
                font = spans[0]["font"]
                is_bold = 1 if "Bold" in font else 0
                y_pos = block["bbox"][1]

                # Predict label using the ML model
                features = [[font_size, is_bold, y_pos]]
                label_encoded = model.predict(features)[0]
                label = label_map_rev[label_encoded]

                if label in ["H1", "H2", "H3"]:
                    outline.append({
                        "level": label,
                        "text": text,
                        "page": page_num
                    })

    return {
        "title": outline[0]["text"] if outline else "Untitled",
        "outline": outline
    }

# 🔽 Run the function
if __name__ == "__main__":
    pdf_path = "C:\\Users\\pande\\OneDrive\\Documents\\ADOBE1A\\InputPDF\\Basic Java Programs.pdf"  # 🔁 Replace with your own PDF path
    result = predict_from_pdf(pdf_path)

    with open("predicted_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    print(" Headings extracted to predicted_output.json")
