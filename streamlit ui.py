# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 18:47:35 2025

@author: igriz
"""
import streamlit as st
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import json
import os

# === CONFIGURATION ===
DATA_PATH = r"C:/Users/igriz/Documents/BOOTCAMP 2024/WEEK 9/DAY1/project-dsml-interactive-travel-planner-main/project-dsml-interactive-travel-planner-main/processed_data_with_embeddings.json"
INDEX_PATH = r"C:/Users/igriz/Documents/BOOTCAMP 2024/WEEK 9/DAY1/project-dsml-interactive-travel-planner-main/project-dsml-interactive-travel-planner-main/vector_store/annoy_index.ann"
EMBEDDING_DIM = 384  # Should match model output

# === Load Model & Data ===
st.title("🗞️ Puerto Rico News & Travel Guide")
st.write("Explore historic newspaper articles and discover interesting places in Puerto Rico!")

@st.cache_resource()
def load_model():
    return SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

@st.cache_resource()
def load_annoy_index():
    index = AnnoyIndex(EMBEDDING_DIM, "angular")
    index.load(INDEX_PATH)
    return index

@st.cache_resource()
def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)

# Load all resources
model = load_model()
index = load_annoy_index()
data = load_data()

# === Search Function ===
def search_articles(query, top_k=5):
    query_embedding = model.encode(query)
    indices, distances = index.get_nns_by_vector(query_embedding, top_k, include_distances=True)

    # Ensure indices are within bounds
    results = [(data[i], dist) for i, dist in zip(indices, distances) if i < len(data)]
    
    return results

# === UI Chatbot ===
user_query = st.text_input("Ask about history, weather, or landmarks in Puerto Rico:")
if user_query:
    st.subheader("🔍 Top Matches:")
    results = search_articles(user_query)
    if results:
        for article, dist in results:
            st.markdown(f"### 📅 {article['date']} - {article['headline']}")
            st.write(f"📰 {article['content'][:300]}...")
            st.write(f"🔹 **Relevance Score:** {round(1 - dist, 3)}")
            st.write("---")
    else:
        st.warning("No relevant articles found. Try a different search!")

st.sidebar.header("⚙️ Filters")
st.sidebar.write("Date range, topic filters coming soon!")
