# movie-recommendation-system
# 🎯 1. Problem Definition 
**Business Problem:** 
Streaming platforms need to recommend relevant movies to users to increase:  
- Watch time
- User engagement 
- Subscription retention

**Why it matters:**  
- 80%+ content consumption on platforms is recommendation-driven 
- Better recommendations = more revenue

# 📊 2. Dataset

**Recommended dataset:**

👉 MovieLens Dataset (open-source)

**Link:**
https://grouplens.org/datasets/movielens/

**Files:**

- movies.csv → movieId, title, genres
- ratings.csv → userId, movieId, rating, timestamp
- tags.csv → userId, movieId, tag

# 🧹 4. Data Preprocessing (src/data_preprocessing.py)
```
import pandas as pd

def load_data():
    movies = pd.read_csv("data/movies.csv")
    ratings = pd.read_csv("data/ratings.csv")
    return movies, ratings

def preprocess(movies, ratings):
    # merge
    df = ratings.merge(movies, on="movieId")

    # drop nulls
    df.dropna(inplace=True)

    return df
```
