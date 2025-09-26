import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib

# ------------------------------
# 1️⃣ Load Features
# ------------------------------
X = np.load("features/X_mfcc.npy")
y = np.load("features/y_labels.npy")

print("Feature matrix shape:", X.shape)
print("Labels shape:", y.shape)

# ------------------------------
# 2️⃣ Train-Test Split
# ------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print("Training samples:", X_train.shape[0])
print("Test samples:", X_test.shape[0])

# ------------------------------
# 3️⃣ Train Classifier
# ------------------------------
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# ------------------------------
# 4️⃣ Evaluate
# ------------------------------
y_pred = clf.predict(X_test)

print("\n=== Test Set Performance ===")
print("Accuracy:", accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred, target_names=["Control", "ALS"]))

# ------------------------------
# 5️⃣ Save Model
# ------------------------------
joblib.dump(clf, "features/als_classifier.pkl")
print("\nTrained model saved as features/als_classifier.pkl")
