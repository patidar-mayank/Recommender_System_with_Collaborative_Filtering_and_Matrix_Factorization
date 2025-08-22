# Recommender System with Collaborative Filtering and Matrix Factorization

This project demonstrates various approaches to building a movie recommender system using collaborative filtering and matrix factorization techniques in Python. The solution is implemented in a Jupyter Notebook and leverages pandas, scikit-learn, and related libraries for data processing and modeling.

## Overview

The goal of this project is to recommend movies to users based on various filtering techniques:

- **Popularity-based recommendation:** Suggests globally popular movies using vote count and average.
- **Content-based filtering:** Uses movie attributes such as genres, cast, director, and overview to recommend similar movies.
- **Matrix factorization:** Applies Singular Value Decomposition (SVD) on a text-based feature matrix to uncover latent factors and recommend similar movies.

## Dataset

The dataset used in this project is a CSV file (`movies.csv`) containing metadata about various movies such as:

- Title
- Genres
- Cast
- Director
- Overview
- Vote average & count
- Other metadata

> **Note:** The notebook expects the dataset at `/content/movies.csv`. Adjust the path as needed for your environment.

## Project Structure

```
.
├── Recommender_System_with_Collaborative_Filtering_and_Matrix_Factorization.ipynb
├── movies.csv
└── README.md
```

## Features & Approaches

### 1. Popularity-Based Recommender (Baseline)

- Computes a weighted rating for each movie using the IMDB formula:
    ```
    Weighted Rating = (v/(v+m)) * R + (m/(v+m)) * C
    ```
    where:
      - v = number of votes for the movie
      - m = minimum votes required (e.g., 85th percentile)
      - R = average rating for the movie
      - C = mean vote across the dataset

- Outputs the top 10 highest-scoring movies as the baseline recommendations.

### 2. Content-Based Filtering (Cosine Similarity)

- Constructs a 'tags' feature by combining genres, cast, and director.
- Applies TF-IDF vectorization to the 'tags' column (max 5000 features).
- Calculates cosine similarity between all movie vectors.
- Includes a recommendation function to return the top 10 most similar movies to a given title.

### 3. Matrix Factorization with SVD

- Builds a bag-of-words feature matrix from the 'overview' column using CountVectorizer.
- Applies Truncated SVD to reduce dimensionality and extract latent features.
- Uses cosine similarity on the SVD-projected vectors to recommend similar movies.

## How to Run

1. **Clone the repository and place your dataset:**
    ```bash
    git clone https://github.com/patidar-mayank/Recommender_System_with_Collaborative_Filtering_and_Matrix_Factorization.git
    cd Recommender_System_with_Collaborative_Filtering_and_Matrix_Factorization
    ```

2. **Ensure you have the following file:**
    - `movies.csv` (movie metadata)

3. **Open the notebook:**
    - Launch Jupyter Notebook or Google Colab.
    - Open `Recommender_System_with_Collaborative_Filtering_and_Matrix_Factorization.ipynb`.

4. **Run all cells in order.**
    - Modify the dataset path if needed.

## Requirements

- Python 3.x
- pandas
- numpy
- scikit-learn
- matplotlib
- (Optional) Jupyter Notebook or Google Colab

Install requirements via pip:
```bash
pip install pandas numpy scikit-learn matplotlib
```


---

**Author:** [patidar-mayank](https://github.com/patidar-mayank)
