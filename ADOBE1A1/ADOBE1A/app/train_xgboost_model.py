import pandas as pd
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

# === Load Dataset ===
df = pd.read_csv("app/FINAL_DATASET_smote.csv")

# === Define Feature Columns ===
features = [
    "font_size", "is_bold", "is_italic", "x0", "x1", "y0", "y1", "y_pos",
    "is_uppercase", "num_words", "text_length",
    "contains_colon_or_dot", "is_numbered_heading", "has_bullets_or_dashes"
]

X = df[features].copy()

# Convert boolean columns to integers
bool_cols = [
    "is_bold", "is_italic", "is_uppercase",
    "contains_colon_or_dot", "is_numbered_heading", "has_bullets_or_dashes"
]
X[bool_cols] = X[bool_cols].astype(int)

# === Encode Labels ===
le = LabelEncoder()
y = le.fit_transform(df["label"])

# === Split Train/Test ===
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# === Train the Model ===
model = XGBClassifier(
    n_estimators=50,
    max_depth=4,
    learning_rate=0.1,
    use_label_encoder=False,  # Avoid deprecation warning
    eval_metric="mlogloss",
    random_state=42
)
model.fit(X_train, y_train)

# === Evaluate ===
y_pred = model.predict(X_test)
print("\n Classification Report:")
print(classification_report(y_test, y_pred, target_names=le.classes_))

# === Save Model and Label Encoder ===
os.makedirs("app/model", exist_ok=True)  # <-- creates directory if missing
joblib.dump(model, "app/model/xgb_model.pkl")
joblib.dump(le, "app/model/label_encoder.pkl")
print(" Model + LabelEncoder saved to: app/model/")
