import joblib
import numpy as np
import pandas as pd

# Load saved matrix and similarity
user_item_matrix, user_similarity = joblib.load("Model/collaborative_model.pkl")

def recommend_products(user_id, top_n=5):
    if user_id not in user_item_matrix.index:
        return f"❌ User '{user_id}' not found."

    user_index = user_item_matrix.index.get_loc(user_id)
    similarity_scores = user_similarity[user_index]
    similar_users_indices = np.argsort(similarity_scores)[::-1][1:6]
    similar_users = user_item_matrix.index[similar_users_indices]

    # Aggregate similar users’ ratings
    recommended_items = user_item_matrix.loc[similar_users].mean(axis=0)

    # Remove already-rated items
    rated_items = user_item_matrix.loc[user_id]
    unrated_items = recommended_items[rated_items == 0]

    # Top N recommendations
    recommendations = unrated_items.sort_values(ascending=False).head(top_n)
    return recommendations

# Example usage
if __name__ == "__main__":
    user = "A123"  # Replace with real user ID from your dataset
    print(recommend_products(user))
