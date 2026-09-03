import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix

from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

# ==========================================
# STEP 1: Load Data & Train/Test Split
# ==========================================
# پروپوزل: پچھلے پروجیکٹ کا صاف شدہ ڈیٹا استعمال کریں
df = pd.read_csv("cleaned_dataset.csv")

# فرضی Fraud ٹارگٹ بنائیں (اگر آپ کے ڈیٹا میں 'OrderStatus' کالم ہے)
# ہم 'Cancelled' یا 'Returned' کو فرضی غیر معمولی/فراڈ کیٹیگری (1) اور باقی کو (0) مان لیتے ہیں
if 'OrderStatus_Cancelled' in df.columns:
    df['is_fraud'] = df['OrderStatus_Cancelled']
elif 'OrderStatus' in df.columns:
    df['is_fraud'] = df['OrderStatus'].apply(lambda x: 1 if x in ['Cancelled', 'Returned'] else 0)
else:
    # اگر کوئی کالم نہیں تو ٹیسٹنگ کے لیے ایک imbalanced کالم بناتے ہیں
    np.random.seed(42)
    df['is_fraud'] = np.random.choice([0, 1], size=len(df), p=[0.95, 0.05])

X = df.drop(columns=['is_fraud', 'OrderID', 'Date', 'CustomerID', 'TrackingNumber', 'ShippingAddress'], errors='ignore')
# صرف numeric کالمز رکھیں
X = X.select_dtypes(include=[np.number])
y = df['is_fraud']

print("=== Target Class Distribution ===")
print(y.value_counts(normalize=True))

# 80/20 Stratified Split (Data Leakage سے بچنے کے لیے SMOTE سے پہلے اسپلٹ کریں)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ==========================================
# STEP 2: Logistic Regression Pipeline + SMOTE
# ==========================================
lr_pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=1000, random_state=42))
])

# Hyperparameter Tuning using GridSearchCV
lr_param_grid = {
    'smote__k_neighbors': [3, 5],
    'classifier__C': [0.1, 1.0]
}

lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
lr_grid.fit(X_train, y_train)

print("\n=== Best Logistic Regression Parameters ===")
print(lr_grid.best_params_)

# ==========================================
# STEP 3: Random Forest Pipeline + SMOTE
# ==========================================
rf_pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(random_state=42))
])

rf_param_grid = {
    'smote__k_neighbors': [3, 5],
    'classifier__n_estimators': [50, 100],
    'classifier__max_depth': [5, 10]
}

rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=3, scoring='roc_auc', n_jobs=-1)
rf_grid.fit(X_train, y_train)

print("\n=== Best Random Forest Parameters ===")
print(rf_grid.best_params_)

# ==========================================
# STEP 4: Evaluation Metrics (Precision, Recall, ROC-AUC)
# ==========================================
def evaluate_model(model, name, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    
    print(f"\n================ {name} Performance ================")
    print(classification_report(y_test, y_pred))
    print(f"ROC-AUC Score: {roc_auc_score(y_test, y_proba):.4f}")
    
    # Confusion Matrix Plot
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'{name} - Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    plt.savefig(f'{name.lower().replace(" ", "_")}_cm.png')
    plt.show()

# دونوں ماڈلز کو جانچیں
evaluate_model(lr_grid.best_estimator_, "Logistic Regression", X_test, y_test)
evaluate_model(rf_grid.best_estimator_, "Random Forest", X_test, y_test)