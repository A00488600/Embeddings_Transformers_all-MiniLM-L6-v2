import pandas as pd
import numpy as np
import json
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# Load old embeddings from JSON
with open("embeddings.json", "r") as f:
    old_embeddings = json.load(f)

# Load the modified classmates file
df_new = pd.read_csv("classmates.csv")

# Load a pre-trained sentence embedding model
model = SentenceTransformer("all-MiniLM-L6-v2")

# Generate new embeddings
df_new["Embedding"] = df_new["Description"].apply(lambda x: model.encode(x).tolist())

# Save new embeddings to a JSON file for future comparisons
new_embeddings = {row["Name"]: row["Embedding"] for _, row in df_new.iterrows()}
with open("embeddings_new.json", "w") as f:
    json.dump(new_embeddings, f, indent=4)

# Compute cosine similarity for modified names
results = []
for name, new_embedding in new_embeddings.items():
    if name in old_embeddings:  # Ensure the name exists in the old embeddings
        old_embedding = np.array(old_embeddings[name])
        new_embedding = np.array(new_embedding)

        similarity = cosine_similarity([old_embedding], [new_embedding])[0][0]
        results.append({"Name": name, "Cosine Similarity": similarity})

# Convert results to DataFrame and print
df_results = pd.DataFrame(results)
print(df_results)

# Save similarity results
df_results.to_csv("cosine_similarity_results.csv", index=False)