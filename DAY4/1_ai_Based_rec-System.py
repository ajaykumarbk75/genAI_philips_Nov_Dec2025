#app.py # this is the FastAPI backend for the AI-driven Recommendation System
#recsys_app.py used for building recommendation system using FastAPI

from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

app = FastAPI(title="AI Recommendation System")

# ----------------------------
# Load data
# ----------------------------
ratings_df = pd.read_csv("ratings.csv")

# Strip whitespace from column names
ratings_df.columns = ratings_df.columns.str.strip()

# Handle duplicates by taking the mean rating for duplicate user-item pairs
ratings_df = ratings_df.groupby(['user_id', 'item_id'], as_index=False)['rating'].mean()

# Create user-item matrix
user_item_matrix = ratings_df.pivot(index='user_id', columns='item_id', values='rating').fillna(0)

# Compute item-item similarity
item_similarity = cosine_similarity(user_item_matrix.T)  # transpose to get items
item_similarity_df = pd.DataFrame(item_similarity, index=user_item_matrix.columns, columns=user_item_matrix.columns)

# ----------------------------
# Pydantic model for API input
# ----------------------------
class RecommendRequest(BaseModel):
    user_id: int
    top_k: int = 3

# ----------------------------
# Recommendation function
# ----------------------------
def recommend_items(user_id, top_k=3):
    if user_id not in user_item_matrix.index:
        return {"error": "User not found"}

    user_ratings = user_item_matrix.loc[user_id]
    scores = {}
    for item in user_item_matrix.columns:
        if user_ratings[item] == 0:  # only recommend unrated items
            similar_items = item_similarity_df[item]
            score = sum(user_ratings[i] * similar_items[i] for i in user_item_matrix.columns)
            scores[item] = score

    # sort by score
    recommended_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
    return [item for item, score in recommended_items]

# ----------------------------
# API Endpoint
# ----------------------------
@app.post("/recommend")
def recommend(req: RecommendRequest):
    recs = recommend_items(req.user_id, req.top_k)
    return {"user_id": req.user_id, "recommendations": recs}

