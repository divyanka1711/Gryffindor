import os
import json
from pathlib import Path
import pandas as pd

def generate_json_output():
    input_csv = Path("app/OUTPUT/PREDICTED_OUTPUT.csv")  #  Correct path to predicted CSV
    output_dir = Path("app/OUTPUT")                      # Output folder for JSON files

    # Ensure output directory exists
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load the predicted CSV
    df = pd.read_csv(input_csv)

    # Ensure required columns exist
    required_cols = {"pdf_file", "text", "page", "label"}
    if not required_cols.issubset(df.columns):
        raise ValueError(f"CSV must contain columns: {required_cols}")

    # Sort the DataFrame
    if "y_pos" in df.columns:
        df = df.sort_values(by=["pdf_file", "page", "y_pos"], ascending=[True, True, False])
    else:
        df = df.sort_values(by=["pdf_file", "page"])

    # Generate JSON per PDF
    for pdf_file, group in df.groupby("pdf_file"):
        title = None
        outline = []

        for _, row in group.iterrows():
            label = row["label"].strip().upper()
            if label == "OTHER":
                continue

            text = str(row["text"]).strip()
            page = int(row["page"])

            if label == "TITLE" and not title:
                title = text
            else:
                outline.append({
                    "level": label,
                    "text": text,
                    "page": page
                })

        if not title:
            title = Path(pdf_file).stem

        output_data = {
            "title": title,
            "outline": outline
        }

        # Save JSON output file
        output_file = output_dir / f"{Path(pdf_file).stem}.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(output_data, f, indent=2, ensure_ascii=False)

        print(f" Processed: {output_file.name}")

if __name__ == "__main__":
    print(" Starting JSON generation...")
    generate_json_output()
    print(" All Done.")
