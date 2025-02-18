# -*- coding: utf-8 -*-
"""
Created on Fri Feb 14 12:14:44 2025

@author: igriz
"""

import json
import os
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
DATA_PATH = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\streamlit\municipalities_landmarks.json"
INDEX_PATH = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\vector_store\municipalities_landmarks.ann"
PROCESSED_DATA_PATH = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\streamlit\processed_municipalities_landmarks.json"
VECTOR_DIM = 384  # Ensure this matches the SentenceTransformer model
NUM_TREES = 10  # Number of trees for Annoy index

# === LOAD MODEL ===
print("🔄 Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("✅ Model loaded!")

# === LOAD DATA ===
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load dataset
data = load_data(DATA_PATH)

# === GENERATE EMBEDDINGS ===
def generate_embeddings(data):
    """Generate embeddings for municipalities and landmarks."""
    for category in ["municipalities", "landmarks"]:
        for entry in data.get(category, []):
            text = f"{entry.get('name', '')} {entry.get('summary', '')}"
            entry["embedding"] = embedding_model.encode(text).tolist()
    return data

print("🔄 Generating embeddings...")
data = generate_embeddings(data)
print("✅ Embeddings generated!")

# === BUILD ANNOY INDEX ===
def build_annoy_index(data):
    index = AnnoyIndex(VECTOR_DIM, "angular")
    all_entries = []  # Store entries for proper indexing
    
    counter = 0
    for category in ["municipalities", "landmarks"]:
        for entry in data.get(category, []):
            if "embedding" in entry:
                index.add_item(counter, entry["embedding"])
                all_entries.append(entry)
                counter += 1
    
    index.build(NUM_TREES)
    index.save(INDEX_PATH)
    return all_entries

print("🔄 Building Annoy index...")
all_entries = build_annoy_index(data)
print(f"✅ Annoy index saved at {INDEX_PATH} with {len(all_entries)} entries!")

# === SAVE PROCESSED DATA ===
with open(PROCESSED_DATA_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ Processed data saved at {PROCESSED_DATA_PATH}")


# %%

import json
import os
from annoy import AnnoyIndex
from sentence_transformers import SentenceTransformer
import streamlit as st

# === CONFIGURATION ===
DATA_PATH = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\streamlit\processed_municipalities_landmarks.json"
INDEX_PATH = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\vector_store\municipalities_landmarks.ann"
VECTOR_DIM = 384  # Ensure this matches the SentenceTransformer model
TOP_K = 5  # Number of results to return

# === LOAD MODEL ===
print("🔄 Loading embedding model...")
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("✅ Model loaded!")

# === LOAD DATA ===
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# Load dataset
data = load_data(DATA_PATH)

# === LOAD ANNOY INDEX ===
index = AnnoyIndex(VECTOR_DIM, "angular")
index.load(INDEX_PATH)

# === SEARCH FUNCTION ===
def search_municipalities_landmarks(query, top_k=TOP_K):
    """Retrieve the most relevant municipalities or landmarks based on the user query."""
    query_embedding = embedding_model.encode(query)
    indices, distances = index.get_nns_by_vector(query_embedding, top_k, include_distances=True)
    
    results = []
    all_entries = data["municipalities"] + data["landmarks"]
    
    for idx, distance in zip(indices, distances):
        if idx < len(all_entries):
            match = all_entries[idx]
            results.append({
                "name": match.get("name", "Unknown"),
                "summary": match.get("summary", "No summary available."),
                "image_url": match.get("image_url", None),
                "distance": round(distance, 3)
            })
    
    return results

# === STREAMLIT UI ===
st.title("🏙️ Explore Puerto Rico's Municipalities & Landmarks")
user_query = st.text_input("🔍 Search for a place or landmark:")

if user_query:
    st.subheader("🔎 Search Results:")
    results = search_municipalities_landmarks(user_query)
    
    if results:
        for place in results:
            st.markdown(f"### 📍 {place['name']}")
            st.write(f"{place['summary']}")
            if place['image_url']:
                st.image(place['image_url'], caption=place['name'], use_column_width=True)
            st.write(f"🔹 **Relevance Score:** {place['distance']}")
            st.write("---")
    else:
        st.warning("No relevant locations found. Try a different search!")

print("✅ Search system ready!")
