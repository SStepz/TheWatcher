from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
from data import df_anime, df_anime_score, anime_svd_model
from database import fetch_real_time_data
from anime_functions import *

# Content-based
tfidf = TfidfVectorizer(stop_words='english')
tfidf_matrix = tfidf.fit_transform((genre for genre in df_anime['genres'].values.astype('U')))
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)

app = FastAPI()

class ContentRequest(BaseModel):
    title: str

class UserRequest(BaseModel):
    userId: str

@app.post("/anime/recommendations/content", tags=['Anime'])
def get_content_based_anime_recommendations(request: ContentRequest):
    title = request.title
    idx = df_anime[df_anime['title'] == title].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    valid_scores = [x for x in sim_scores if df_anime.iloc[x[0]]['score'] != -1]
    sorted_scores = sorted(valid_scores, key=lambda x: (x[1], df_anime.iloc[x[0]]['score']), reverse=True)
    top_animes = [x for x in sorted_scores if x[0] != idx][:10]
    recommended_indices = [x[0] for x in top_animes]
    recommendations = df_anime.iloc[recommended_indices][['title', 'genres', 'score']]
    return recommendations.to_dict(orient='records')

@app.post("/anime/recommendations/user", tags=['Anime'])
def get_hybrid_user_based_anime_recommendations(request: UserRequest):
    user_id = request.userId
    df_real_time = fetch_real_time_data('anime')
    user_anime_matrix, user_anime_sparse_matrix = get_user_item_anime_matrix(df_anime_score, df_real_time)
    knn = get_anime_knn_model(user_anime_sparse_matrix)
    recommended_animes_ids = hybrid_anime_recommendations(user_id, anime_svd_model, knn, user_anime_matrix, user_anime_sparse_matrix)
    recommendations = get_anime_details_by_ids(recommended_animes_ids)
    return recommendations.to_dict(orient='records')