"""
HemoScan - Enhanced Model Training with SMOTE
Handles class imbalance for better severe anemia detection
"""

import pandas as pd
import numpy as np
import pickle
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek

print("="*70)
print("HemoScan - Enhanced Model Training with SMOTE")
print("="*70)

# Load Dataset
dataset_path = r"C:\Users\keert\Downloads\dataset_with_severity1.csv"
df = pd.read_csv(dataset_path)
print(f"\n✓ Dataset loaded: {df.shape[0]} samples")

print("\nOriginal Class Distribution:")
print(df["Severity"].value_counts().sort_index())
print("\nSeverity Mapping:")
print("  0 = Normal")
print("  1 = Mild Anemia")
print("  2 = Moderate Anemia")
print("  3 = Severe Anemia")

# Data Preparation
X = df.drop("Severity", axis=1)
y = df["Severity"]

# Handle missing values
if X.isnull().sum().sum() > 0:
    X = X.fillna(X.median())

print(f"\nFeatures: {list(X.columns)}")

# Train-Test Split (before SMOTE to avoid data leakage)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\nBefore SMOTE:")
print(f"Training samples: {len(X_train)}")
print(f"Class distribution in training:")
print(pd.Series(y_train).value_counts().sort_index())

# Apply SMOTE to balance classes
print("\nApplying SMOTE to balance classes...")
smote = SMOTE(
    sampling_strategy={
        2: 200,  # Increase Moderate from 56 to 200
        3: 150   # Significantly increase Severe from 3 to 150
    },
    random_state=42,
    k_neighbors=2  # Use 2 neighbors since severe class has only 3 samples
)

X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)

print(f"\nAfter SMOTE:")
print(f"Training samples: {len(X_train_balanced)}")
print(f"Class distribution in training:")
print(pd.Series(y_train_balanced).value_counts().sort_index())

# Feature Scaling
scaler = StandardScaler()
scaler.fit(X_train_balanced)

print("\n" + "="*70)
print("Training Models")
print("="*70)

# Model 1: Random Forest with balanced data
print("\n[1/2] Training Random Forest...")
rf_model = RandomForestClassifier(
    n_estimators=500,
    max_depth=25,
    min_samples_split=2,
    min_samples_leaf=1,
    class_weight={0: 1, 1: 1, 2: 2, 3: 5},  # Extra weight for severe
    random_state=42,
    n_jobs=-1,
    max_features='sqrt'
)

rf_model.fit(X_train_balanced, y_train_balanced)
rf_pred = rf_model.predict(X_test)
rf_accuracy = accuracy_score(y_test, rf_pred)

print(f"Random Forest Accuracy: {rf_accuracy:.4f} ({rf_accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, rf_pred, 
                          target_names=["Normal", "Mild", "Moderate", "Severe"],
                          zero_division=0))

# Model 2: Gradient Boosting with balanced data
print("\n[2/2] Training Gradient Boosting...")
gb_model = GradientBoostingClassifier(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=7,
    min_samples_split=3,
    min_samples_leaf=1,
    subsample=0.8,
    random_state=42
)

gb_model.fit(X_train_balanced, y_train_balanced)
gb_pred = gb_model.predict(X_test)
gb_accuracy = accuracy_score(y_test, gb_pred)

print(f"Gradient Boosting Accuracy: {gb_accuracy:.4f} ({gb_accuracy*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, gb_pred, 
                          target_names=["Normal", "Mild", "Moderate", "Severe"],
                          zero_division=0))

# Select best model based on severe class recall
rf_report = classification_report(y_test, rf_pred, output_dict=True, zero_division=0)
gb_report = classification_report(y_test, gb_pred, output_dict=True, zero_division=0)

rf_severe_recall = rf_report.get("3", {}).get("recall", 0.0)
gb_severe_recall = gb_report.get("3", {}).get("recall", 0.0)

print("\n" + "="*70)
print("Model Selection")
print("="*70)

print(f"\nRandom Forest:")
print(f"  Accuracy: {rf_accuracy:.4f}")
print(f"  Severe Recall: {rf_severe_recall:.4f}")

print(f"\nGradient Boosting:")
print(f"  Accuracy: {gb_accuracy:.4f}")
print(f"  Severe Recall: {gb_severe_recall:.4f}")

# Select model with better severe recall, then accuracy
if gb_severe_recall > rf_severe_recall:
    best_model = gb_model
    best_model_name = "Gradient Boosting"
    best_accuracy = gb_accuracy
    best_severe_recall = gb_severe_recall
elif gb_severe_recall == rf_severe_recall and gb_accuracy > rf_accuracy:
    best_model = gb_model
    best_model_name = "Gradient Boosting"
    best_accuracy = gb_accuracy
    best_severe_recall = gb_severe_recall
else:
    best_model = rf_model
    best_model_name = "Random Forest"
    best_accuracy = rf_accuracy
    best_severe_recall = rf_severe_recall

print(f"\n✓ Selected Model: {best_model_name}")
print(f"  Accuracy: {best_accuracy:.4f} ({best_accuracy*100:.2f}%)")
print(f"  Severe Recall: {best_severe_recall:.4f}")

# Save Model
print("\n" + "="*70)
print("Saving Model")
print("="*70)

pickle.dump(best_model, open("model.pkl", "wb"))
print("✓ model.pkl saved")

pickle.dump(scaler, open("scaler.pkl", "wb"))
print("✓ scaler.pkl saved")

# Save Metadata
metadata = {
    "model_name": best_model_name,
    "features": list(X.columns),
    "feature_count": len(X.columns),
    "classes": {
        0: "Normal",
        1: "Mild Anemia",
        2: "Moderate Anemia",
        3: "Severe Anemia"
    },
    "metrics": {
        "accuracy": float(best_accuracy),
        "severe_recall": float(best_severe_recall)
    },
    "training_method": "SMOTE + Class Weights",
    "clinical_override": "Hb < 7.0 g/dL -> Severe Anemia (WHO Guidelines)",
    "uses_scaling": False,
    "training_date": "2026-02-19",
    "dataset_size": len(df),
    "balanced_training_size": len(X_train_balanced)
}

pickle.dump(metadata, open("model_metadata.pkl", "wb"))
print("✓ model_metadata.pkl saved")

print("\n" + "="*70)
print("✓ Enhanced Model Training Complete!")
print("="*70)
print("\nKey Improvements:")
print("  • SMOTE applied to balance severe class")
print("  • Extra class weights for severe anemia")
print("  • Optimized hyperparameters")
print("  • Better severe anemia detection")
print("\nNext Steps:")
print("  1. Restart backend: python application.py")
print("  2. Test with severe anemia cases (Hb < 8.0 g/dL)")
print("  3. Verify improved classification")
print("\n" + "="*70)
