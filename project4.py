import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.corpus import wordnet

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# NLTK کے ضروری پیکجز ڈاؤن لوڈ کریں (بشمول اپ ڈیٹ شدہ tagger)
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# ==========================================
# STEP 1: Sample Unstructured Text Dataset
# ==========================================
data = {
    'review': [
        "This product is amazing! I absolutely love it.",
        "Terrible quality, broke after two days. Do not buy!",
        "Great value for money, highly recommended.",
        "Worst experience ever, totally disappointed.",
        "Very useful and easy to operate. Excellent buy.",
        "Waste of money. Extremely bad service.",
        "Satisfied with the purchase, works fine.",
        "Horrible customer care and damaged product.",
        "Superb build quality and fast shipping!",
        "Defective item, completely useless."
    ],
    'sentiment': [1, 0, 1, 0, 1, 0, 1, 0, 1, 0]  # 1 = Positive, 0 = Negative
}

df = pd.DataFrame(data)

# ==========================================
# STEP 2: Strict Text Pre-Processing Pipeline
# ==========================================
lemmatizer = WordNetLemmatizer()

# Stop-words کی لسٹ سے 'not' جیسے نفی والے الفاظ برقرار رکھیں
stop_words = set(stopwords.words('english')) - {'not', 'nor', 'no'}

def get_wordnet_pos(word):
    """WordNet Lemmatizer کے لیے Part-of-Speech Tag نکالنا"""
    try:
        tag = nltk.pos_tag([word])[0][1][0].upper()
    except Exception:
        tag = 'N'
    tag_dict = {"J": wordnet.ADJ, "N": wordnet.NOUN, "V": wordnet.VERB, "R": wordnet.ADV}
    return tag_dict.get(tag, wordnet.NOUN)

def clean_text(text):
    # 1. Lowercasing & Special Characters ہٹانا
    text = re.sub(r'[^a-zA-Z\s]', '', text.lower())
    words = text.split()
    
    # 2. Stop-words removal & POS-Guided Lemmatization
    cleaned = [
        lemmatizer.lemmatize(w, get_wordnet_pos(w))
        for w in words if w not in stop_words
    ]
    return " ".join(cleaned)

df['cleaned_review'] = df['review'].apply(clean_text)

print("=== Raw vs Cleaned Reviews ===")
print(df[['review', 'cleaned_review']].head())

# ==========================================
# STEP 3: TF-IDF Vectorization & Train/Test Split
# ==========================================
X = df['cleaned_review']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# TF-IDF conversion (Unigrams + Bigrams)
tfidf = TfidfVectorizer(ngram_range=(1, 2), max_features=1000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

print("\n=== TF-IDF Matrix Shape ===")
print("Train Shape:", X_train_tfidf.shape)

# ==========================================
# STEP 4: Multinomial Naive Bayes Model Training
# ==========================================
model = MultinomialNB(alpha=1.0)  # Laplace smoothing
model.fit(X_train_tfidf, y_train)

# ==========================================
# STEP 5: Evaluation & Testing
# ==========================================
y_pred = model.predict(X_test_tfidf)

print("\n================ Evaluation Report ================")
print(f"Accuracy: {accuracy_score(y_test, y_pred) * 100:.2f}%")
print(classification_report(y_test, y_pred))

# ==========================================
# STEP 6: Live Prediction System
# ==========================================
def predict_sentiment(new_review):
    cleaned = clean_text(new_review)
    vec = tfidf.transform([cleaned])
    pred = model.predict(vec)[0]
    result = "POSITIVE 🙂" if pred == 1 else "NEGATIVE 🙁"
    print(f"Review: '{new_review}' ---> Sentiment: {result}")

print("\n=== Live Review Predictions ===")
predict_sentiment("The product is exceptionally good and reliable.")
predict_sentiment("Not good at all, completely broken.")