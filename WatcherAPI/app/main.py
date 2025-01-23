from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np
import gc
from sklearn.feature_extraction.text import TfidfVectorizer
from app.data import df_anime, df_anime_score, anime_svd_model, df_movie, df_movie_score, movie_svd_model, df_tv, df_tv_score, tv_svd_model
import faiss
from app.database import fetch_real_time_data
from app.anime_functions import *
from app.movie_functions import *
from app.tv_functions import *

# Content-based anime recommendations
anime_tfidf = TfidfVectorizer(stop_words='english')
anime_tfidf_matrix = anime_tfidf.fit_transform(df_anime['content'].values.astype('U')).astype(np.float32)
anime_tfidf_matrix_dense = anime_tfidf_matrix.toarray()
epsilon = 1e-10
anime_tfidf_matrix_dense = anime_tfidf_matrix_dense / (np.linalg.norm(anime_tfidf_matrix_dense, axis=1, keepdims=True) + epsilon)
anime_index = faiss.IndexFlatIP(anime_tfidf_matrix_dense.shape[1])
anime_index.add(anime_tfidf_matrix_dense)

# Content-based movie recommendations
movie_tfidf = TfidfVectorizer(stop_words='english')
movie_tfidf_matrix = movie_tfidf.fit_transform(df_movie['genres'].values.astype('U')).astype(np.float32)
movie_tfidf_matrix_dense = movie_tfidf_matrix.toarray()
epsilon = 1e-10
movie_tfidf_matrix_dense = movie_tfidf_matrix_dense / (np.linalg.norm(movie_tfidf_matrix_dense, axis=1, keepdims=True) + epsilon)
movie_index = faiss.IndexFlatIP(movie_tfidf_matrix_dense.shape[1])
movie_index.add(movie_tfidf_matrix_dense)

# Content-based tv series recommendations
tv_tfidf = TfidfVectorizer(stop_words='english')
tv_tfidf_matrix = tv_tfidf.fit_transform(df_tv['genres'].values.astype('U')).astype(np.float32)
tv_tfidf_matrix_dense = tv_tfidf_matrix.toarray()
epsilon = 1e-10
tv_tfidf_matrix_dense = tv_tfidf_matrix_dense / (np.linalg.norm(tv_tfidf_matrix_dense, axis=1, keepdims=True) + epsilon)
tv_index = faiss.IndexFlatIP(tv_tfidf_matrix_dense.shape[1])
tv_index.add(tv_tfidf_matrix_dense)

app = FastAPI()

origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ContentRequest(BaseModel):
    mediaId: int
    number: int

class UserRequest(BaseModel):
    userId: str
    number: int

@app.post("/anime/recommendations/content", tags=['Anime'])
def get_content_based_anime_recommendations(request: ContentRequest):
    mediaId = request.mediaId
    n = request.number
    idx = df_anime[df_anime['id'] == mediaId].index[0]
    anime_vector = anime_tfidf_matrix_dense[idx].reshape(1, -1).astype(np.float32)
    distances, indices = anime_index.search(anime_vector, n+1)
    indices = indices[0]
    distances = distances[0]
    filtered_indices = [(i, d) for i, d in zip(indices, distances) if i != idx]
    filtered_indices = sorted(filtered_indices, key=lambda x: (x[1], df_anime.iloc[x[0]]['score']), reverse=True)
    top_indices = [i for i, _ in filtered_indices[:n]]
    recommended_indices = [i for i in top_indices if df_anime.iloc[i]['score'] != -1]
    recommendations = df_anime.iloc[recommended_indices][['id', 'title', 'genres', 'score', 'image_url']]
    recommendations = recommendations.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    recommendations = recommendations.infer_objects(copy=False)
    return recommendations.to_dict(orient='records')

@app.post("/anime/recommendations/user", tags=['Anime'])
def get_hybrid_user_based_anime_recommendations(request: UserRequest):
    user_id = request.userId
    n = request.number
    df_real_time = fetch_real_time_data('anime')
    user_anime_matrix, user_anime_sparse_matrix = get_user_item_anime_matrix(df_anime_score, df_real_time)

    del df_real_time
    gc.collect()

    knn = get_anime_knn_model(user_anime_sparse_matrix)
    recommended_animes_ids = hybrid_anime_recommendations(user_id, anime_svd_model, knn, user_anime_matrix, user_anime_sparse_matrix, n)
    recommendations = get_anime_details_by_ids(recommended_animes_ids)

    del user_anime_matrix, user_anime_sparse_matrix, knn, recommended_animes_ids
    gc.collect()

    return recommendations.to_dict(orient='records')

@app.post("/movie/recommendations/content", tags=['Movie'])
def get_content_based_movie_recommendations(request: ContentRequest):
    mediaId = request.mediaId
    n = request.number
    idx = df_movie[df_movie['id'] == mediaId].index[0]
    movie_vector = movie_tfidf_matrix_dense[idx].reshape(1, -1).astype(np.float32)
    distances, indices = movie_index.search(movie_vector, n+1)
    indices = indices[0]
    distances = distances[0]
    filtered_indices = [(i, d) for i, d in zip(indices, distances) if i != idx]
    filtered_indices = sorted(filtered_indices, key=lambda x: (x[1], df_movie.iloc[x[0]]['vote_average']), reverse=True)
    top_indices = [i for i, _ in filtered_indices[:n]]
    recommended_indices = [i for i in top_indices if df_movie.iloc[i]['vote_average'] != -1]
    recommendations = df_movie.iloc[recommended_indices][['id', 'title', 'genres', 'vote_average', 'release_date', 'poster_path']]
    recommendations = recommendations.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    recommendations = recommendations.infer_objects(copy=False)
    return recommendations.to_dict(orient='records')

@app.post("/movie/recommendations/user", tags=['Movie'])
def get_hybrid_user_based_movie_recommendations(request: UserRequest):
    user_id = request.userId
    n = request.number
    df_real_time = fetch_real_time_data('movie')
    user_movie_matrix, user_movie_sparse_matrix = get_user_item_movie_matrix(df_movie_score, df_real_time)

    del df_real_time
    gc.collect()

    knn = get_movie_knn_model(user_movie_sparse_matrix)
    recommended_movies_ids = hybrid_movie_recommendations(user_id, movie_svd_model, knn, user_movie_matrix, user_movie_sparse_matrix, n)
    recommendations = get_movie_details_by_ids(recommended_movies_ids)

    del user_movie_matrix, user_movie_sparse_matrix, knn, recommended_movies_ids
    gc.collect()

    return recommendations.to_dict(orient='records')

@app.post("/tv/recommendations/content", tags=['TV Series'])
def get_content_based_tv_recommendations(request: ContentRequest):
    mediaId = request.mediaId
    n = request.number
    idx = df_tv[df_tv['id'] == mediaId].index[0]
    tv_vector = tv_tfidf_matrix_dense[idx].reshape(1, -1).astype(np.float32)
    distances, indices = tv_index.search(tv_vector, n+1)
    indices = indices[0]
    distances = distances[0]
    filtered_indices = [(i, d) for i, d in zip(indices, distances) if i != idx]
    filtered_indices = sorted(filtered_indices, key=lambda x: (x[1], df_tv.iloc[x[0]]['vote_average']), reverse=True)
    top_indices = [i for i, _ in filtered_indices[:n]]
    recommended_indices = [i for i in top_indices if df_tv.iloc[i]['vote_average'] != -1]
    recommendations = df_tv.iloc[recommended_indices][['id', 'name', 'genres', 'vote_average', 'first_air_date', 'poster_path']]
    recommendations = recommendations.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    recommendations = recommendations.infer_objects(copy=False)
    return recommendations.to_dict(orient='records')

@app.post("/tv/recommendations/user", tags=['TV Series'])
def get_hybrid_user_based_tv_recommendations(request: UserRequest):
    user_id = request.userId
    n = request.number
    df_real_time = fetch_real_time_data('tv')
    user_tv_matrix, user_tv_sparse_matrix = get_user_item_tv_matrix(df_tv_score, df_real_time)

    del df_real_time
    gc.collect()

    knn = get_tv_knn_model(user_tv_sparse_matrix)
    recommended_tvs_ids = hybrid_tv_recommendations(user_id, tv_svd_model, knn, user_tv_matrix, user_tv_sparse_matrix, n)
    recommendations = get_tv_details_by_ids(recommended_tvs_ids)

    del user_tv_matrix, user_tv_sparse_matrix, knn, recommended_tvs_ids
    gc.collect()

    return recommendations.to_dict(orient='records')