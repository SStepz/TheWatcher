import numpy as np
import pandas as pd
from joblib import load

# Anime Data
df_anime = pd.read_csv('../Dataset/Anime/anime.csv')
df_anime_score = pd.read_excel('../Dataset/Anime/users-scores.xlsx')
df_anime_score['user_id'] = df_anime_score['user_id'].astype('str')
anime_svd_model = load('../Model/anime_svd_model.joblib')

anime_scores = df_anime['score'][df_anime['score'] != -1]
anime_scores = anime_scores.astype('float')
score_mean = round(anime_scores.mean() , 2)
df_anime['score'] = df_anime['score'].replace(-1, score_mean)
df_anime['score'] = df_anime['score'].astype('float64')
df_anime['rank'] = df_anime['rank'].replace(-1, np.nan)
df_anime['rank'] = df_anime['rank'].astype('float64')
df_anime_score = df_anime_score[df_anime_score['username'].notnull()]

# Movie Data
df_movie = pd.read_csv('../Dataset/Movie/movie.csv')
df_movie_score = pd.read_excel('../Dataset/Movie/movie_user_ratings.xlsx')
df_movie_score['userId'] = df_movie_score['userId'].astype('str')
# movie_svd_model = load('../Model/movie_svd_model.joblib')

df_movie = df_movie.drop(columns=['status', 'release_date', 'revenue', 'runtime', 'adult', 'backdrop_path', 'budget', 'homepage', 'imdb_id', 'original_language', 'original_title', 'overview', 'poster_path', 'tagline', 'production_companies', 'production_countries', 'spoken_languages'])
df_movie = df_movie[df_movie['vote_average'] != 0]
df_movie_score = df_movie_score[df_movie_score['userId'].notnull()]

# TV Data
df_tv = pd.read_csv('../Dataset/TV/tv.csv')
# df_tv_score = pd.read_excel('../Dataset/TV/tv_user_ratings.xlsx')
# tv_svd_model = load('../Model/tv_svd_model.joblib')

df_tv = df_tv.drop(columns=['original_language', 'overview', 'adult', 'backdrop_path', 'first_air_date', 'last_air_date', 'homepage', 'original_name', 'poster_path', 'status', 'tagline', 'created_by', 'languages', 'networks', 'origin_country', 'spoken_languages', 'production_companies', 'production_countries', 'episode_run_time'])
df_tv = df_tv[df_tv['vote_average'] != 0]