# 🛍️ E-commerce Product Recommendation System (Hybrid)

This project is a hybrid recommendation engine for e-commerce platforms that uses both Content-Based and Collaborative Filtering approaches to suggest products to users.

---

## 🔍 Project Features

- ✅ **Content-Based Filtering** using TF-IDF on product metadata (`product name`, `category`, `about product`)
- ✅ **Collaborative Filtering** using Surprise SVD based on `user_id`, `product_id`, and `rating`
- ✅ **Streamlit Web App** to interact with the system
- ✅ Shows product image and link from dataset

---

## 🗂 Folder Structure

EcommerceRecommendation_Hybrid/
├── Data/
│ └── amazon.csv # Dataset
├── Model/
│ └── content_model.pkl # Trained content model
│ └── collaborative_model.pkl # Trained collaborative model
├── main_content.py # Train TF-IDF model
├── main_collab.py # Train collaborative model
├── app.py # Streamlit interface
├── requirements.txt
└── README.md