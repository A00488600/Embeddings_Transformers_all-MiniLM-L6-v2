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

# Initialize the all-mpnet-base-v2 model
model_mpnet = SentenceTransformer('all-mpnet-base-v2')

# Generate embeddings with mpnet
mpnet_embeddings = {}
for _, row in df.iterrows():
    name = row['Name']
    text = row['Description']
    embedding = model_mpnet.encode(text)
    mpnet_embeddings[name] = embedding.tolist()

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

# Geting name #17 is Anuja
your_name = df['Name'].iloc[17] 

# Get rankings from both models
minilm_rankings = get_rankings(your_name, minilm_embeddings)
mpnet_rankings = get_rankings(your_name, mpnet_embeddings)

# Calculate Spearman correlation
rho, p_value = stats.spearmanr(
    [minilm_rankings.index(name) for name in minilm_rankings],
    [mpnet_rankings.index(name) for name in minilm_rankings]
)

print(f"Spearman's rho: {rho:.3f}")
print(f"p-value: {p_value:.3f}")

# Print rankings side by side with indices
print("\nRankings comparison:")
comparison = pd.DataFrame({
    'MiniLM': minilm_rankings,
    'MPNet': mpnet_rankings
})

# Add original indices
minilm_indices = [df[df['Name'] == name].index[0] for name in minilm_rankings]
mpnet_indices = [df[df['Name'] == name].index[0] for name in mpnet_rankings]

comparison['MiniLM_Index'] = minilm_indices
comparison['MPNet_Index'] = mpnet_indices

# Reorder columns to show indices next to names
comparison = comparison[['MiniLM', 'MiniLM_Index', 'MPNet', 'MPNet_Index']]
print(comparison)