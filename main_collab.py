import pandas as pd
import numpy as np
import joblib
from sklearn.metrics.pairwise import cosine_similarity

# Load dataset
df = pd.read_csv("Data/amazon.csv")

# Drop rows with missing user_id, product_id, or rating
df = df.dropna(subset=["user_id", "product_id", "rating"])

# Create a pivot table (user-item matrix)
user_item_matrix = df.pivot_table(index="user_id", columns="product_id", values="rating").fillna(0)

# Compute cosine similarity between users
user_similarity = cosine_similarity(user_item_matrix)

# Save the user-item matrix and similarity matrix for later use
joblib.dump((user_item_matrix, user_similarity), "Model/collaborative_model.pkl")
print("✅ Collaborative filtering model saved.")
