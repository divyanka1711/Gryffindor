import pandas as pd
import joblib

def run_prediction():
    # === CONFIG ===
    INPUT_CSV = "app/INPUT/input_unlabeled.csv"
    OUTPUT_CSV = "app/PREDICTED_OUTPUT.csv"

    # === Load Trained Model and LabelEncoder ===
    model = joblib.load("app/model/xgb_model.pkl")
    le = joblib.load("app/model/label_encoder.pkl")

    # === Load Input Data ===
    df = pd.read_csv(INPUT_CSV)

    # === Define Features ===
    features = [
        "font_size", "is_bold", "is_italic", "x0", "x1", "y0", "y1", "y_pos",
        "is_uppercase", "num_words", "text_length",
        "contains_colon_or_dot", "is_numbered_heading", "has_bullets_or_dashes"
    ]

    # === Ensure Boolean Columns Are Integer (0/1) ===
    bool_cols = [
        "is_bold", "is_italic", "is_uppercase",
        "contains_colon_or_dot", "is_numbered_heading", "has_bullets_or_dashes"
    ]
    df[bool_cols] = df[bool_cols].astype(int)

    # === Run Model Prediction ===
    X = df[features]
    preds = model.predict(X)
    df["label"] = le.inverse_transform(preds)

    # === Save Predicted Output CSV ===
    df[["text", "page", "pdf_file", "label"]].to_csv(OUTPUT_CSV, index=False)

    print(f" Predictions saved to: {OUTPUT_CSV}")

# Optional: Run directly for debugging
if __name__ == "__main__":
    run_prediction()
