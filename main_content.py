import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import os

df = pd.read_csv("Data/amazon.csv")

# Combine product features
df["combined"] = df["product_name"].fillna('') + " " + df["category"].fillna('') + " " + df["about_product"].fillna('')

# TF-IDF vectorization
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform(df["combined"])

# Cosine similarity
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

# Save model
os.makedirs("Model", exist_ok=True)
joblib.dump((df, cosine_sim), "Model/content_model.pkl")
print("✅ Content-based model saved.")
