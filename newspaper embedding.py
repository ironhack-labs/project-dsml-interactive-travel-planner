# -*- coding: utf-8 -*-
"""
Created on Thu Feb 13 14:36:48 2025

@author: igriz
"""

import json
import numpy as np
from annoy import AnnoyIndex
import os
from sentence_transformers import SentenceTransformer
# === CONFIGURATION ===
VECTOR_STORE_PATH = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\vector_store\new_index"
DATA_PATH = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\final data.json"
VECTOR_DIM = 384  # Ensure this matches your embeddings' dimensions
NUM_TREES = 10  # Number of trees for AnnoyIndex (higher = more accuracy, slower indexing)
# Load pre-trained sentence transformer model
embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# === LOAD DATA ===
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_data(data, file_path):
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

# === GENERATE EMBEDDINGS ===
def generate_embeddings(data):
    """Generate embeddings for newspaper content."""
    for article in data.get("newspapers", []):
        text = f"{article.get('headline', '')} {article.get('content', '')}"
        article["embedding"] = embedding_model.encode(text).tolist()
    return data

# === BUILD ANNOY INDEX ===
def build_annoy_index(embeddings):
    index = AnnoyIndex(VECTOR_DIM, 'angular')
    for i, vector in enumerate(embeddings):
        index.add_item(i, vector)
    index.build(NUM_TREES)
    index.save(VECTOR_STORE_PATH)
    print(f"✅ Annoy index saved at {VECTOR_STORE_PATH}")

# === MAIN EXECUTION ===
if __name__ == "__main__":
    print("🔄 Loading data...")
    data = load_data(DATA_PATH)

    # ✅ Generate embeddings if missing
    if "embedding" not in data["newspapers"][0]:  
        print("🔄 Generating embeddings...")
        data = generate_embeddings(data)
        save_data(data, DATA_PATH)
        print("✅ Embeddings saved to JSON!")

    # ✅ Extract embeddings
    embeddings = [entry["embedding"] for entry in data.get("newspapers", []) if "embedding" in entry]

    if not embeddings:
        print("🚨 No embeddings found after generation!")
    else:
        print(f"✅ Found {len(embeddings)} embeddings. Building Annoy index...")
        build_annoy_index(embeddings)

# %%

# Path where we'll save the data with embeddings
EMBEDDING_JSON_PATH = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\processed_data_with_embeddings.json"

# Load original newspaper data
with open("final data.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Inject embeddings into the newspaper articles
for i, entry in enumerate(data.get("newspapers", [])):
    if i < len(embeddings):  # Ensure valid index
        entry["embedding"] = embeddings[i]

# Save the new JSON file with embeddings included
with open(EMBEDDING_JSON_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=4, ensure_ascii=False)

print(f"✅ Embeddings saved in {EMBEDDING_JSON_PATH}")

# %%


# === CONFIGURATION ===
DATA_FILE = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\final data.json"
EMBEDDINGS_FILE = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\processed_data_with_embeddings.json"
ANNOY_INDEX_FILE = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\vector_store\annoy_index"
VECTOR_SIZE = 384  # Dimension of the embedding model
TOP_K = 5  # Number of results to return

# === LOAD MODEL ===
print("🔄 Loading embedding model...")
model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
print("✅ Model loaded!")

# === LOAD DATA ===
def load_data(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

# === CHECK IF EMBEDDINGS ALREADY EXIST ===
if os.path.exists(EMBEDDINGS_FILE):
    print("✅ Found existing embeddings. Loading...")
    with open(EMBEDDINGS_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)
else:
    print("🔄 Generating embeddings...")
    data = load_data(DATA_FILE)
    
    for article in data["newspapers"]:
        article["embedding"] = model.encode(article["content"]).tolist()  # Generate embeddings
    
    with open(EMBEDDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
    print("✅ Embeddings saved to JSON!")

# === BUILD ANNOY INDEX ===
print(f"✅ Found {len(data['newspapers'])} embeddings. Building Annoy index...")
index = AnnoyIndex(VECTOR_SIZE, 'angular')

for i, article in enumerate(data["newspapers"]):
    index.add_item(i, np.array(article["embedding"]))

index.build(10)  # 10 trees for efficiency
# Ensure the directory for the Annoy index exists
VECTOR_STORE_DIR = "vector_store"
os.makedirs(VECTOR_STORE_DIR, exist_ok=True)  # ✅ Creates directory if it doesn’t exist

# Save Annoy index with full valid path
ANNOY_INDEX_FILE = os.path.join(VECTOR_STORE_DIR, "annoy_index.ann")  # ✅ Ensure proper file extension
index.save(ANNOY_INDEX_FILE)  # ✅ Saves correctly
print(f"✅ Annoy index saved at {ANNOY_INDEX_FILE}")

# === SEARCH FUNCTION WITH KEYWORD FILTER ===
def search_articles(query, index, model, data, top_k=5):
    query_vector = model.encode([query])[0]
    nearest_neighbors = index.get_nns_by_vector(query_vector, top_k, include_distances=True)

    results = []
    for idx, dist in zip(*nearest_neighbors):
        article = data["newspapers"][idx]
        headline = article["headline"]
        content = article["content"]

        # ✅ **Filter Out Irrelevant Results**
        keywords = ["hurricane", "storm", "damage", "destruction", "flood", "disaster", "wind"]
        if any(word in content.lower() for word in keywords):
            results.append({
                "headline": headline,
                "date": article["date"],
                "snippet": content[:500],  # First 500 characters for preview
                "distance": round(dist, 2)
            })

    return results

# === RUN SEARCH TEST ===
query = "Hurricane damage in Puerto Rico"
print(f"\n🔍 **Query:** {query}\n")

results = search_articles(query, index, model, data, TOP_K)

if results:
    for i, res in enumerate(results):
        print(f"\n➡️ **Match {i+1}: {res['headline']}** (Distance: {res['distance']})")
        print(f"📅 **Date:** {res['date']}")
        print(f"📰 **Snippet:** {res['snippet'][:500]}...")
        print("-" * 80)
else:
    print("❌ No relevant results found!")

            
# %%

from annoy import AnnoyIndex
import json
from sentence_transformers import SentenceTransformer

# === CONFIGURATION ===
DATA_PATH = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\processed_data_with_embeddings.json"
INDEX_PATH = r"C:\Users\igriz\Documents\BOOTCAMP 2024\WEEK 9\DAY1\project-dsml-interactive-travel-planner-main\project-dsml-interactive-travel-planner-main\vector_store\annoy_index.ann"
EMBEDDING_DIM = 384  # Ensure this matches the SentenceTransformer model

# === LOAD DATA ===
with open(DATA_PATH, "r", encoding="utf-8") as f:
    full_data = json.load(f)

# Extract only the newspapers section
newspaper_data = full_data.get("newspapers", [])

if not newspaper_data:
    raise ValueError("❌ No newspaper data found in the dataset!")

# Ensure embeddings exist
missing_embeddings = [entry for entry in newspaper_data if "embedding" not in entry or not isinstance(entry["embedding"], list)]

if missing_embeddings:
    raise ValueError(f"❌ Some newspaper entries are missing embeddings! ({len(missing_embeddings)} out of {len(newspaper_data)})")

print(f"✅ Found {len(newspaper_data)} newspaper articles with embeddings!")

# === BUILD & SAVE ANNOY INDEX ===
index = AnnoyIndex(EMBEDDING_DIM, "angular")

for i, entry in enumerate(newspaper_data):
    index.add_item(i, entry["embedding"])

index.build(10)  # 10 trees for a good balance of speed & accuracy
index.save(INDEX_PATH)

print(f"✅ Annoy index saved at {INDEX_PATH}")


# === SEARCH FUNCTION ===
def retrieve_relevant_documents(query, top_k=5):
    """
    Retrieve the most relevant newspaper articles based on the user query.
    """
    # Load Annoy index
    index.load(INDEX_PATH)

    # Load the embedding model
    embedding_model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

    # Convert query to embedding
    query_embedding = embedding_model.encode(query)

    # Search Annoy index
    indices = index.get_nns_by_vector(query_embedding, top_k, include_distances=True)

    # Retrieve top matches
    results = []
    for idx, distance in zip(*indices):
        if idx < len(newspaper_data):
            match = newspaper_data[idx]
            results.append({
                "headline": match["headline"],
                "date": match["date"],
                "snippet": match["content"][:300],  # Show first 300 characters
                "distance": round(distance, 3)
            })

    return results


# === TEST SEARCH ===
if __name__ == "__main__":
    query = "Hurricane damage in Puerto Rico"
    top_matches = retrieve_relevant_documents(query)

    if top_matches:
        print("\n✅ Top Matches Found:")
        for match in top_matches:
            print(f"\n➡️ **{match['headline']}** ({match['date']})\n📌 Distance: {match['distance']}\n📰 {match['snippet']}...\n" + "-"*80)
    else:
        print("\n❌ No matches found!")
