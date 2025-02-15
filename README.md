# Embedding Matchmaking

_"Words can't describe how unique your interests are... but coordinates can" - Sean Ashley, circa 2023_

A flattened embedding space of names clustered based on their interests using the sentence-transformers all-MiniLM-L6-v2 model. Created for the UW Startups S23 Kickoff event with guidance from [Jacky Zhao](https://jzhao.xyz/) and [Sean Ashley](https://www.linkedin.com/in/sean-ashley). [Simha Kalimipalli](https://github.com/Simha-Kalimipalli) later aded interactivity!

![Sample output of script](https://github.com/nERD8932/EmbeddingsAssignment/blob/main/visualization.png?raw=true)

## What Are Embeddings and why do we need them?

While using language as a means to communicate is intrinsic to humans, the same is not true for machines. Machines, at the core of their operations, run on 0s and 1s (aka bits). This means that for tasks that require machines to "understand" the complex patterns and relations that exist in languages, we need to represent those words mathematically, that is, with numbers.

This is where embeddings come in.

Embeddings are a way to represent words, sentences, and concepts as vectors in an n-dimentional space.

To put it simply, we can imagine embeddings as players in a football field. In the field, we try to place players that play well together closer to each other, and players that play worse when together farther away from each other. Now while this may seem simple, imagine if our field had over a million players (the number of words in the English language) and if the players existed in many different dimensions (other than the usual 3). You can see how this might get complicated.

## Understanding with an Example
| Name | What are your interests? (or varying permutations of this question) |
| -------- | ------- |
| Somto | I enjoy reading, cycling, playing chess, and story-based video games (think Red Dead Redemption, Baldur's Gate, GTA). |
| Samir | I enjoy playing games like Elden Ring, Legend of Zelda and God of War. |
| Drira | I’m passionate about hiking, reading, meditation, movies, and embracing new challenges. |

Here, we can see that Somto and Samir have the common interests of "games", Driar and Somto have the common interest of "reading" and Samir and Driar don't have a common interest. Based on these interests, we can predict that when embedding, we would place Somto and Samir close, Somto and Driar close but place Samir and Driar far from each other.

This is exactly what we see in our sample visualization:

![Example Visual](https://github.com/nERD8932/EmbeddingsAssignment/blob/main/example.png?raw=true)

As we can see, through embedding we can map the complex relations of language into an abstract n-dimensional mathematical space. Embeddings are used today for most natural language processing tasks, including LLMs like ChatGPT.

# Embedding Sensitivity Tests
The results below show that the rankings produced by different embedding models are moderately correlated but not identical.

![Output of Sens Test](https://github.com/nERD8932/EmbeddingsAssignment/blob/main/model_test_output.png?raw=true)

The ****Spearman’s rank correlation coefficient (𝜌) is 0.407**** with a ****p-value of 0.084****, indicating a weak to moderate positive correlation between the rankings generated by **`all-MiniLM-L6-v2`** and **`all-mpnet-base-v2`**.

### **Quantitative Considerations**

- The **𝜌 value of 0.407** suggests a weak to moderate correlation, meaning that while both models capture some similarities in ranking, their orderings are not strongly aligned.
- A value closer to 1 would indicate a high degree of agreement, while a value near 0 would suggest randomness. The 0.407 result implies that different embeddings lead to noticeable shifts in ranking order.
- Some names are ranked closely across both models, while others experience significant shifts, demonstrating the impact of embedding differences on perceived similarity.

### **Qualitative Considerations**

- The top-ranked classmate differs between models (**MiniLM ranks Max Zhao as the closest, while MPNet ranks Louise Fear first**).
- Several classmates experience shifts in ranking. For instance, **Somto Muotoe is ranked 5th in MiniLM but 2nd in MPNet**, indicating that embeddings from different models may prioritize other aspects of similarity.
- Although there is partial agreement in rankings, the models differ in how they weigh contextual relationships, affecting the nearest neighbors identified.

These findings indicate that **model choice significantly affects ranking results**. While there is some alignment between the two models, notable differences suggest that **embedding spaces encode relationships differently**, leading to varied similarity rankings. This highlights the importance of model selection in applications where ranking consistency matters.

Dimension Reduction Analysis

1. The UMAP algorithm is sufficient enough for our purpose of identifying classmates who have similar interests.
points which are nearer have more similar while which are farther are more different

2. Changing the seed changed how the visualization looked, it is because as we change the seed the state where the visualization starts to reduce the dimensions changed

3. Out of n_neighbors, min_dist, n_components, metric we chose to tune only n_neighbors(2, 20), min_dist(0, 0.99), metric(['cosine', 'euclidean']) lower n_neighbors concentrate on very local structure of the vectors greater n_neighbors values will push UMAP to look at the global structure of the vectors cause getting a n_components value which is greater than 2 for which we get a maximum Spearman's rank correlation coefficient is not a good way of visualizing it in 2D cause those relations are translated into 2D in that case so i fixed to 2 to have a better visualization

4. used optuna for hyperparameter search to maximize Spearman's rank correlation coefficient

the original implementation visualization had changed completely by change in random seed but the tuned implementation has a significant amount of changes in the visualizations by the change of random seed so the tuned implementation is more stable towards change in random seed

Moderate to Good Preservation of Structure: Model retains a moderate to good preservation of relationships, with a Spearman correlation of ~0.64. This suggests that, while not perfect, the global structure is largely maintained which has the patterns in the visualization.

Euclidean Metric for Normalized Embeddings: The Euclidean metric is well-suited for normalized embeddings, preserving relative distances and providing a meaningful representation of data relationships.

Loss of Pairwise Relationships: Model loses some pairwise relationships, with correlation falling below 0.75. This results in the loss of finer data nuances, affecting tasks requiring precise understanding of individual data points.

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
