import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from joblib import dump

# Anime Data
df_anime = pd.read_csv('../Dataset/Anime/anime.csv')
df_anime_score = pd.read_excel('../Dataset/Anime/users-scores.xlsx')
df_anime_score['user_id'] = df_anime_score['user_id'].astype('str')

anime_scores = df_anime['score'][df_anime['score'] != -1]
anime_scores = anime_scores.astype('float')
score_mean = round(anime_scores.mean() , 2)
df_anime['score'] = df_anime['score'].replace(-1, score_mean)
df_anime['score'] = df_anime['score'].astype('float64')
df_anime['rank'] = df_anime['rank'].replace(-1, np.nan)
df_anime['rank'] = df_anime['rank'].astype('float64')
df_anime_score = df_anime_score[df_anime_score['username'].notnull()]

# Model Training
reader = Reader(rating_scale=(df_anime_score['rating'].min(), df_anime_score['rating'].max()))
data = Dataset.load_from_df(df_anime_score[['user_id', 'anime_id', 'rating']], reader)

# Define the SVD model
svd_model = SVD()

# Evaluate the model using cross-validation
cross_validate(svd_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the SVD model on the entire dataset
svd_trainset = data.build_full_trainset()
svd_model.fit(svd_trainset)

dump(svd_model, '../Model/anime_svd_model.joblib')