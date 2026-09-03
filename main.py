import pandas as pd
import numpy as np
from sklearn.impute import KNNImputer

# ==========================================
# STEP 2: Load and Inspect Data
# ==========================================
# (یہاں اپنی فائل کا صحیح نام دیں، مثلاً dataset.csv یا Dataset for Data Analytics.csv)
df = pd.read_csv("dataset.csv")

print("=== Data Info ===")
df.info()

print("\n=== Missing Values Before Imputation ===")
print(df.isnull().sum())


# ==========================================
# STEP 3: Handle Missing Data (Statistical Imputation)
# ==========================================
# 1. Missingness percentage معلوم کرنا
missing_pct = (df.isnull().sum() / len(df)) * 100

# 2. Imputation Strategies لاگو کرنا
for col in df.columns:
    pct = missing_pct[col]
    if 0 < pct <= 5:
        # 5% سے کم خالی خانوں کی لائنیں ختم کرنا
        df.dropna(subset=[col], inplace=True)
    elif 5 < pct <= 20:
        # 5% سے 20% میں Median یا Mode سے بھرنا
        if df[col].dtype in ['int64', 'float64']:
            df[col].fillna(df[col].median(), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    elif pct > 20:
        # 20% سے زیادہ میں KNN Imputer لگانا
        if df[col].dtype in ['int64', 'float64']:
            imputer = KNNImputer(n_neighbors=5)
            df[[col]] = imputer.fit_transform(df[[col]])

print("\n=== Missing Values After Imputation ===")
print(df.isnull().sum())

# ==========================================
# STEP 4: Identify & Neutralize Outliers
# ==========================================
def cap_outliers_iqr(data, column):
    Q1 = data[column].quantile(0.25)
    Q3 = data[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # حد سے باہر کی ویلیوز کو Boundaries پر کیپ کرنا
    data[column] = np.clip(data[column], lower_bound, upper_bound)
    return data

# عددی (Numeric) کالمز پر لاگو کریں
numeric_cols = ['Quantity', 'UnitPrice', 'TotalPrice']
for col in numeric_cols:
    if col in df.columns:
        df = cap_outliers_iqr(df, col)

print("\n=== Outliers Neutralized ===")
print(df[numeric_cols].describe())

# ==========================================
# STEP 5: Feature Engineering (3 New Features)
# ==========================================
# 1. Has_Coupon (1 اگر کوپن تھا، 0 اگر نہیں تھا)
df['Has_Coupon'] = df['CouponCode'].notnull().astype(int)

# 2. Order_Month (تاریخ سے مہینہ الگ کرنا)
df['Date'] = pd.to_datetime(df['Date'])
df['Order_Month'] = df['Date'].dt.month

# 3. Avg_Item_Cart_Price (فی سامان اوسط قیمت)
df['Avg_Item_Cart_Price'] = df['TotalPrice'] / (df['ItemsInCart'] + 1e-5)

print("\n=== New Features Created ===")
print(df[['Has_Coupon', 'Order_Month', 'Avg_Item_Cart_Price']].head())

# ==========================================
# STEP 6: Categorical Encoding & Correlation Check
# ==========================================
# Text variables کو Number (0/1) میں تبدیل کرنا
categorical_cols = ['PaymentMethod', 'OrderStatus', 'ReferralSource']
df_encoded = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# 0.80 سے زیادہ ملتی جلتی features (Multicollinearity) ختم کرنا
corr_matrix = df_encoded.corr(numeric_only=True).abs()
upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > 0.80)]
df_final = df_encoded.drop(columns=to_drop)

# صاف شدہ فائنل ڈیٹا کو نئی CSV میں محفوظ کریں
df_final.to_csv("cleaned_dataset.csv", index=False)

print("\n=== Cleaned Dataset Ready & Saved ===")
print("Final Dataset Shape:", df_final.shape)