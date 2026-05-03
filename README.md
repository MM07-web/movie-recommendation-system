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

,,,
 #  movie-recommendation-system/
│
├── data/
│   ├── movies.csv
│   ├── ratings.csv
│
├── notebooks/
│   └── eda.ipynb
│
├── src/
│   ├── data_preprocessing.py
│   ├── feature_engineering.py
│   ├── model.py
│   ├── recommend.py
│
├── models/
│   └── similarity.pkl
│
├── outputs/
│   └── recommendations.csv
│
├── app.py
├── main.py
├── requirements.txt
├── README.md
,,,

