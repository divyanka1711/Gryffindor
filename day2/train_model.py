# train_model.py

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

# 1. Load labeled data
df = pd.read_csv("labeled_lines_sample.csv")

# 2. Convert 'is_bold' to 0/1 if needed
df["is_bold"] = df["is_bold"].astype(int)

# 3. Encode labels to numbers
label_map = {"H1": 0, "H2": 1, "H3": 2, "TEXT": 3}
df["label_encoded"] = df["label"].map(label_map)

# 4. Split features and target
X = df[["font_size", "is_bold", "y_pos"]]
y = df["label_encoded"]

# 5. Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 6. Train model
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# 7. Save model
dump(model, "heading_model.joblib")

# 8. Evaluate
acc = model.score(X_test, y_test)
print(f" Model trained. Accuracy: {acc:.2f}")
