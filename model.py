import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import pickle
import os

# Load datasets
true_df = pd.read_csv("data/True.csv")
fake_df = pd.read_csv("data/Fake.csv")

# Label: 1 = Real, 0 = Fake
true_df['label'] = 1
fake_df['label'] = 0

# Combine & shuffle
data = pd.concat([true_df, fake_df], ignore_index=True)
data = data.sample(frac=1).reset_index(drop=True)

# Drop nulls
data = data[['title', 'text', 'label']].dropna()

# ===============================
# 1. Model using ONLY the title
# ===============================
X_title = data['title']
y = data['label']

X_train_t, X_test_t, y_train_t, y_test_t = train_test_split(X_title, y, test_size=0.2, random_state=42)

vectorizer_title = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_title_vec = vectorizer_title.fit_transform(X_train_t)
X_test_title_vec = vectorizer_title.transform(X_test_t)

model_title = LogisticRegression()
model_title.fit(X_train_title_vec, y_train_t)
title_accuracy = accuracy_score(y_test_t, model_title.predict(X_test_title_vec))
print(f"[Title Model] Accuracy: {title_accuracy:.4f}")

# Save title model
with open("models/model_title.pkl", "wb") as f:
    pickle.dump(model_title, f)

with open("models/vectorizer_title.pkl", "wb") as f:
    pickle.dump(vectorizer_title, f)

# ===============================
# 2. Model using title + text
# ===============================
data['content'] = data['title'] + " " + data['text']
X_content = data['content']

X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(X_content, y, test_size=0.2, random_state=42)

vectorizer_content = TfidfVectorizer(stop_words='english', max_df=0.7)
X_train_content_vec = vectorizer_content.fit_transform(X_train_c)
X_test_content_vec = vectorizer_content.transform(X_test_c)

model_content = LogisticRegression()
model_content.fit(X_train_content_vec, y_train_c)
content_accuracy = accuracy_score(y_test_c, model_content.predict(X_test_content_vec))
print(f"[Content Model] Accuracy: {content_accuracy:.4f}")

# Save content model
with open("models/model_content.pkl", "wb") as f:
    pickle.dump(model_content, f)

with open("models/vectorizer_content.pkl", "wb") as f:
    pickle.dump(vectorizer_content, f)
