1. the extent to which the UMAP algorithm is adequate for our purpose of 
identifying classmates who have similar interests. 

2. changing the seed changed how the visualization looked, it is because as we change the seed the state where the visualization starts to reduce the dimensions changed

3. out of n_neighbors, min_dist, n_components, metric 
we chose to tune only n_neighbors(2, 20), min_dist(0, 0.99), metric(['cosine', 'euclidean'])
lower n_neighbors concentrate on very local structure of the vectors
greater n_neighbors values will push UMAP to look at the global structure of the vectors
cause getting a n_components value which is greater than 2 for which we get a maximum Spearman's rank correlation coefficient is not a good way of visualizing it in 2D cause those relations are translated into 2D in that case so i fixed to 2 to have a better visualization

4. used optuna for hyperparameter search to maximize Spearman's rank correlation coefficient 

5. the original implementation visualization had changed completely by change in random seed but the tuned implementation has a significant amount of changes in the visualizations by the change of random seed so the tuned implementation is more stable towards change in random seed

6. Moderate to Good Preservation of Structure: Model retains a moderate to good preservation of relationships, with a Spearman correlation of ~0.64. This suggests that, while not perfect, the global structure is largely maintained which has the patterns in the visualization.

Euclidean Metric for Normalized Embeddings: The Euclidean metric is well-suited for normalized embeddings, preserving relative distances and providing a meaningful representation of data relationships.

Loss of Pairwise Relationships: Model loses some pairwise relationships, with correlation falling below 0.75. This results in the loss of finer data nuances, affecting tasks requiring precise understanding of individual data points.
