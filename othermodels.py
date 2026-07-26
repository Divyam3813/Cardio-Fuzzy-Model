import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, recall_score
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# ==============================================================================
# 1. LOAD & CLEAN LOCAL DATASET
# ==============================================================================
# Read local file (handling common missing value indicators like '?')
df = pd.read_csv("heart.csv", na_values="?").dropna().drop_duplicates()

# Ensure target & feature columns are clean numeric values
feature_cols = ['cp', 'thalach', 'oldpeak', 'exang', 'ca', 'thal']
target_col = 'target'

for col in feature_cols + [target_col]:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df = df.dropna()

# Convert target to binary (0 = No Disease, 1 = Disease present)
df[target_col] = (df[target_col] > 0).astype(int)

X = df[feature_cols]
y = df[target_col]

# ==============================================================================
# 2. TRAIN-TEST SPLIT & FEATURE SCALING
# ==============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

scaler = StandardScaler()
X_tr_sc = scaler.fit_transform(X_train)
X_te_sc = scaler.transform(X_test)

# ==============================================================================
# 3. MODEL TRAINING & BENCHMARKING
# ==============================================================================
ann = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42).fit(X_tr_sc, y_train)
rf = RandomForestClassifier(n_estimators=100, max_depth=4, random_state=42).fit(X_tr_sc, y_train)
lr = LogisticRegression(random_state=42).fit(X_tr_sc, y_train)

models = {
    "Deep ANN": ann, 
    "Random Forest": rf, 
    "Logistic Regression": lr
}

print(f"\n--- EVALUATION ON UNSEEN TEST SET ({len(y_test)} Patients) ---")
for name, model in models.items():
    preds = model.predict(X_te_sc)
    acc = accuracy_score(y_test, preds) * 100
    sens = recall_score(y_test, preds) * 100
    print(f"{name:20s} | Accuracy: {acc:.2f}% | Sensitivity (Recall): {sens:.2f}%")