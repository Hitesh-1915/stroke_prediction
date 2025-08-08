# -*- coding: utf-8 -*-
"""
Created on Mon Jun 30 12:18:09 2025

@author: donad
"""

# BRAIN STROKE PREDICTION AND DIAGNOSIS
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, auc
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
sns.set(style="whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)

# Step 1: Define column names
column_names = [
    'gender', 'age', 'hypertension', 'heart_disease', 'ever_married',
    'work_type', 'Residence_type', 'avg_glucose_level', 'bmi',
    'smoking_status', 'stroke'
]

# Step 2: Load the data
df = pd.read_csv(r'C:\Users\donad\OneDrive\Desktop\brain_stroke.csv')

# Step 3: Convert numeric columns
df['age'] = pd.to_numeric(df['age'], errors='coerce')
df['avg_glucose_level'] = pd.to_numeric(df['avg_glucose_level'], errors='coerce')
df['bmi'] = pd.to_numeric(df['bmi'], errors='coerce')

# Step 4: Handle missing values
df['bmi'].fillna(df['bmi'].median(), inplace=True)
df['age'].fillna(df['age'].median(), inplace=True)
df['avg_glucose_level'].fillna(df['avg_glucose_level'].median(), inplace=True)

for col in ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']:
    df[col].fillna('Unknown', inplace=True)

# Step 5: Encode categorical variables
label_encoders = {}
categorical_cols = ['gender', 'ever_married', 'work_type', 'Residence_type', 'smoking_status']

for col in categorical_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
    label_encoders[col] = le

# Step 6: Define features and target
X = df.drop('stroke', axis=1)
y = df['stroke']

# Step 7: Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 8: Train Random Forest model
rf_model = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
rf_model.fit(X_train, y_train)

# Step 9: Predictions
y_pred = rf_model.predict(X_test)
y_pred_proba = rf_model.predict_proba(X_test)[:, 1]  # Probability for class 1

# ================== PLOT 1: Confusion Matrix ==================
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# ================== PLOT 2: Feature Importance ==================
plt.figure(figsize=(10, 6))
importances = rf_model.feature_importances_
indices = np.argsort(importances)[::-1]
plt.title('Feature Importance in Stroke Prediction')
plt.bar(range(X.shape[1]), importances[indices], align='center', color='skyblue')
plt.xticks(range(X.shape[1]), [X.columns[i] for i in indices], rotation=45)
plt.tight_layout()
plt.show()


# Step 10: Print Evaluation Metrics
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
