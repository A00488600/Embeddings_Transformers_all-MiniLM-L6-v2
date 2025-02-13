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

Here, we can see that Somto and Samir have the common interests of "games", Sriar and Somto have the common interest of "reading" and Samir and Driar don't have a common interest. Based on these interests, we can predict that when embedding, we would place Somto and Samir close, Somto and Driar close but place Samir and Driar far from each other.

This is exactly what we see in our sample visualization:

![Example Visual](https://github.com/nERD8932/EmbeddingsAssignment/blob/main/example.png?raw=true)

As we can see, through embedding we can map the complex relations of language into an abstract n-dimensional mathematical space. Embeddings are used today for most natural language processing tasks, including LLMs like ChatGPT.

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
