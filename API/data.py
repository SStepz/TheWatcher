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

# TV Data