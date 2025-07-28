import pandas as pd
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE

# Load your labeled dataset
df = pd.read_csv("C:\\Users\\pande\\OneDrive\\Documents\\ADOBE1A\\app\\FINAL_DATASET.csv")

# Define features
features = [
    "font_size", "is_bold", "is_italic", "x0", "x1", "y0", "y1", "y_pos",
    "is_uppercase", "num_words", "text_length",
    "contains_colon_or_dot", "is_numbered_heading", "has_bullets_or_dashes"
]

# Extract features and convert booleans
X = df[features].copy()
bool_cols = ["is_bold", "is_italic", "is_uppercase", "contains_colon_or_dot", "is_numbered_heading", "has_bullets_or_dashes"]
X[bool_cols] = X[bool_cols].astype(int)

# Encode labels
le = LabelEncoder()
y = le.fit_transform(df["label"])

# Apply SMOTE
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X, y)

# Rebuild the balanced DataFrame
df_resampled = pd.DataFrame(X_res, columns=features)
df_resampled["label"] = le.inverse_transform(y_res)

# Save to new CSV
df_resampled.to_csv("FINAL_DATASET_smote.csv", index=False)
print(" New SMOTE-balanced dataset saved as 'FINAL_DATASET_smote.csv'")
