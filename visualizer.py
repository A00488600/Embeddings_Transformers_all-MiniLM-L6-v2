import csv
import umap
import optuna
import matplotlib.pyplot as plt
from scipy import spatial, stats
from sklearn.preprocessing import StandardScaler
from sentence_transformers import SentenceTransformer
import numpy as np
import json
import os

project_path = str(os.getcwd()) + "\\"

# Read attendees and their responses from a CSV file
attendees_map = {}
with open(project_path + 'classmates.csv', newline='') as csvfile:
    attendees = csv.reader(csvfile, delimiter=',', quotechar='"')
    next(attendees)  # Skip the header row
    for row in attendees:
        name, paragraph = row
        attendees_map[paragraph] = name

# Generate sentence embeddings
model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
paragraphs = list(attendees_map.keys())
embeddings = model.encode(paragraphs)

# Create a dictionary to store embeddings for each person
person_embeddings = {attendees_map[paragraph]: embedding for paragraph, embedding in zip(paragraphs, embeddings)}

# Save person embeddings to a file
embeddings_path = project_path + "embeddings.json"
person_embeddings_serializable = {
    person: embedding.tolist() for person, embedding in person_embeddings.items()
}
with open(embeddings_path, "w") as json_file:
    json.dump(person_embeddings_serializable, json_file)
print(f"Embeddings saved successfully to {embeddings_path}")

# Function to optimize UMAP parameters
def objective(trial):
    n_neighbors = trial.suggest_int('n_neighbors', 2, 20)
    min_dist = trial.suggest_float('min_dist', 0.0, 0.99)
    metric = trial.suggest_categorical('metric', ['cosine', 'euclidean'])

    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, n_components=2, metric=metric, random_state=42)
    
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(list(person_embeddings.values()))
    reduced_data = reducer.fit_transform(scaled_data)
    
    # Compute pairwise distances in original embedding space (cosine similarity)
    original_distances = spatial.distance.pdist(embeddings, metric='cosine')
    reduced_distances = spatial.distance.pdist(reduced_data, metric='euclidean')
    
    # Compute Spearman rank correlation
    spearman_corr, _ = stats.spearmanr(original_distances, reduced_distances)
    return spearman_corr

# Run Optuna hyperparameter optimization
study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=80)

# Best hyperparameters
best_params = study.best_params
print("Best Hyperparameters:", best_params)

# Apply best UMAP parameters
best_reducer = umap.UMAP(n_neighbors=best_params['n_neighbors'], min_dist=best_params['min_dist'], n_components=2, metric=best_params['metric'], random_state=42)
scaler = StandardScaler()
scaled_data = scaler.fit_transform(list(person_embeddings.values()))
final_reduced_data = best_reducer.fit_transform(scaled_data)

# Visualization
plt.figure(figsize=(10, 6))
plt.scatter(final_reduced_data[:, 0], final_reduced_data[:, 1], alpha=0.7)
for i, name in enumerate(attendees_map.values()):
    plt.annotate(name, (final_reduced_data[i, 0], final_reduced_data[i, 1]), fontsize=8)
plt.title("UMAP Visualization of Sentence Embeddings")
plt.xlabel("Component 1")
plt.ylabel("Component 2")
plt.show()
