import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from scipy import stats
from scipy.spatial.distance import cosine

# Load existing embeddings from all-MiniLM-L6-v2
with open("embeddings.json", "r") as f:
    minilm_embeddings = json.load(f)

# Read the classmates data
df = pd.read_csv("classmates.csv")

# Initialize the second model - let's use BERT base
model_bert = SentenceTransformer('bert-base-nli-mean-tokens')

# Generate embeddings with BERT
bert_embeddings = {}
for _, row in df.iterrows():
    name = row['Name']
    text = row['Description']
    embedding = model_bert.encode(text)
    bert_embeddings[name] = embedding.tolist()

# Function to get rankings for a target person
def get_rankings(target_name, embeddings_dict):
    target_embedding = np.array(embeddings_dict[target_name])
    distances = []
    
    for name, embedding in embeddings_dict.items():
        if name != target_name:
            dist = cosine(target_embedding, np.array(embedding))
            distances.append((name, dist))
    
    # Sort by distance and extract just the names in order
    return [x[0] for x in sorted(distances, key=lambda x: x[1])]

# Get your name from the CSV (assuming first row is you)
your_name = df['Name'].iloc[0]

# Get rankings from both models
minilm_rankings = get_rankings(your_name, minilm_embeddings)
bert_rankings = get_rankings(your_name, bert_embeddings)

# Calculate Spearman correlation
rho, p_value = stats.spearmanr(
    [minilm_rankings.index(name) for name in minilm_rankings],
    [bert_rankings.index(name) for name in minilm_rankings]
)

print(f"Spearman's rho: {rho:.3f}")
print(f"p-value: {p_value:.3f}")

# Print rankings side by side
print("\nRankings comparison:")
comparison = pd.DataFrame({
    'MiniLM': minilm_rankings,
    'BERT': bert_rankings
})
print(comparison)