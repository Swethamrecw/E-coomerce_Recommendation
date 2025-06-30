import streamlit as st
import pandas as pd
import joblib
def set_background(image_file):
    with open(image_file, "rb") as img_file:
        encoded = base64.b64encode(img_file.read()).decode()
    background_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded}");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }}
    </style>
    """

# Load trained content-based model
df, cosine_sim = joblib.load("Model/content_model.pkl")

st.title("🛍️ Hybrid E-commerce Product Recommender")

# User selects product
product_list = df['product_name'].dropna().unique().tolist()
selected_product = st.selectbox("Choose a product:", product_list)

# Show recommendations
if st.button("Recommend Similar Products"):
    try:
        idx = df[df["product_name"] == selected_product].index[0]
        sim_scores = list(enumerate(cosine_sim[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:6]

        st.subheader("Top Recommendations:")
        for i, score in sim_scores:
            st.markdown(f"**{df.iloc[i]['product_name']}**")
            st.write(f"📦 Category: {df.iloc[i]['category']} | ⭐ Rating: {df.iloc[i]['rating']}")
            if pd.notna(df.iloc[i]['img_link']):
                st.image(df.iloc[i]['img_link'], width=150)
            if pd.notna(df.iloc[i]['product_link']):
                st.markdown(f"[🔗 View Product]({df.iloc[i]['product_link']})")
    except Exception as e:
        st.error(f"Error: {e}")

