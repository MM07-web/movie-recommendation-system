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

  # 📁 3. Project Structure
```
movie-recommendation-system/
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
 ```


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
# 📊 5. EDA (notebooks/eda.ipynb)

```
import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("data/ratings.csv")

# rating distribution
df['rating'].hist()
plt.title("Rating Distribution")
plt.show()

# most rated movies
top_movies = df['movieId'].value_counts().head(10)
top_movies.plot(kind='bar')
plt.title("Top Rated Movies")
plt.show()
```

# ⚙️ 6. Feature Engineering (src/feature_engineering.py)
```
import pandas as pd

def create_user_item_matrix(df):
    user_item = df.pivot_table(index='userId', columns='title', values='rating')
    return user_item.fillna(0)
```

# 🤖 7. Model Building (src/model.py)

```
from sklearn.metrics.pairwise import cosine_similarity
import pickle

def train_model(user_item):
    similarity = cosine_similarity(user_item)
    
    with open("models/similarity.pkl", "wb") as f:
        pickle.dump(similarity, f)

    return similarity
```
 # ML Model ( - Predict Ratings)
 ```
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

def train_ml_model(df):
    X = df[['userId', 'movieId']]
    y = df['rating']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model = RandomForestRegressor()
    model.fit(X_train, y_train)

    return model
```

# 📈 8. Model Evaluation
```
from sklearn.metrics import mean_squared_error

pred = model.predict(X_test)
rmse = mean_squared_error(y_test, pred, squared=False)
print("RMSE:", rmse)

```

# ⚡ 9. Model Optimization
```
from sklearn.model_selection import GridSearchCV

params = {
    'n_estimators': [50, 100]
}

grid = GridSearchCV(RandomForestRegressor(), params)
grid.fit(X_train, y_train)

print(grid.best_params_)

```
# 💾 11. Save Model
```
import joblib
joblib.dump(model, "models/rf_model.pkl")

```
# 🎬 12. Recommendation Engine (src/recommend.py)
```
import pickle
import pandas as pd

def recommend(movie_name, user_item):
    similarity = pickle.load(open("models/similarity.pkl", "rb"))
    
    index = user_item.columns.get_loc(movie_name)
    scores = list(enumerate(similarity[index]))
    
    sorted_scores = sorted(scores, key=lambda x: x[1], reverse=True)[1:6]
    
    recommendations = [user_item.columns[i[0]] for i in sorted_scores]
    return recommendations
```

# 🌐 13. Deployment (Flask API → app.py)
```
from flask import Flask, request, jsonify
import pandas as pd
from src.recommend import recommend

app = Flask(__name__)

user_item = pd.read_csv("data/user_item.csv")

@app.route("/recommend", methods=["GET"])
def get_recommendation():
    movie = request.args.get("movie")
    result = recommend(movie, user_item)
    return jsonify(result)

if __name__ == "__main__":
    app.run(debug=True)
```
# ▶️ main.py
```
from src.data_preprocessing import load_data, preprocess
from src.feature_engineering import create_user_item_matrix
from src.model import train_model

movies, ratings = load_data()
df = preprocess(movies, ratings)

user_item = create_user_item_matrix(df)
train_model(user_item)
```

# 📦 requirements.txt
```
pandas
numpy
scikit-learn
matplotlib
flask
joblib
```



# 🎬 Movie Recommendation System

## Overview
A personalized movie recommendation system using collaborative filtering and machine learning.

## Tech Stack
Python, Pandas, Scikit-learn, Flask

## Features
- Recommend movies based on similarity
- ML-based rating prediction
- REST API support

## How to Run
pip install -r requirements.txt
python main.py
python app.py

## Use Case
Streaming platforms like Netflix, Amazon Prime

## Limitations
- Cold start problem
- No deep learning model

## Future Roadmap
- Add deep learning (Neural CF)
- Add user interface


