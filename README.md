# Embedding Matchmaking

_"Words can't describe how unique your interests are... but coordinates can" - Sean Ashley, circa 2023_

A flattened embedding space of names clustered based on their interests using the sentence-transformers all-MiniLM-L6-v2 model.

![Sample output of script](https://github.com/ansonyuu/matchmaking/blob/main/sample.png?raw=true)

# Component - Data Analysis!
We made slight modifications to the interests in the classmates dataset, primarily by refining word choices and rephrasing certain parts.

|                |Old Desc|New Desc| Cosine Similarity |
|----------------|-------------------------------|-----------------------------|-----------------------------------------|
|Anuja Gamage|`'I like playing MMOs and experimenting with new AI models'`            |'I like playing MMORPGs and trying out emerging AI models'            |0.883|
|Sriram Ramesh          |`'I like Competitive coding, playing soccer, ping pong and pool'`            |'I love Competitive programming, playing football, table tennis, but hate pool'            |0.753|
|Samir Amin Sheikh|`'I enjoy playing games like Elden Ring, Legend of Zelda and God of War'`|'I obsessively play games such as dark souls, Legend of Zelda and God of War'|0.824|


1.  **Anuja Gamage** – Changed “MMOs” to “MMORPGs” and “experimenting with new AI models” to “trying out emerging AI models.”
2.  **Sriram Ramesh** – Replaced “Competitive coding” with “Competitive programming,” changed “soccer, ping pong” to “football, table tennis,” and added **“but hate pool.”**
3.  **Samir Amin Sheikh** – Reworded “I enjoy playing” to **“I obsessively play”** and replaced “Elden Ring” with “Dark Souls.”

These changes resulted in **lower cosine similarity scores**, especially for **Sriram Ramesh (0.75)** and **Samir Amin Sheikh (0.82)**, while **Anuja Gamage’s similarity (0.88) remained unchanged**.

The **large drop in similarity for Sriram** is likely due to the addition of **negative sentiment** (“but hate pool”), which introduces a different contextual meaning. Similarly, for **Samir**, the phrase **“I obsessively play”** conveys a much stronger emotion than “I enjoy playing,” causing a shift in the embedding.

For **Anuja**, the score remained stable because the changes retained the original meaning. The replacement of “MMOs” with “MMORPGs” is a **minor specificity adjustment**, and rewording the AI phrase does not drastically alter the semantic representation.

In summary, **minor wording changes have a small impact, but introducing new sentiments (like strong emotions or negation) significantly alters embeddings, leading to lower similarity scores**.



## Instructions for use

1. Collect or format your data in the following format

| Name  | What are your interests? (or varying permutations of this question) |
| ----- | ------------------------------------------------------------------- |
| Alice | I love being the universal placeholder for every CS joke ever       |
| Bob   | I too love being the universal placeholder for every CS joke        |

2. Clone the repository
3. Install all required packages using pip or conda:

- `umap-learn`
- `scikit-learn`
- `scipy`
- `sentence-transformers`
- `matplotlib`
- `pyvis`
- `pandas`
- `numpy`
- `seaborn`
- `branca`

4. Replace `attendees.csv` in `visualizer.ipynb` with the path to your downloaded data
5. Run all cells
6. Bask in the glory of having an awesome new poster
7. Make two (!) cool interactive visualizations
