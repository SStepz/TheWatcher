import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from joblib import dump

# Movie Data
df_movie = pd.read_csv('../Dataset/Movie/movie.csv')
df_movie = df_movie.drop(columns=['status', 'release_date', 'revenue', 'runtime', 'adult', 'backdrop_path', 'budget', 'homepage', 'imdb_id', 'original_language', 'original_title', 'overview', 'poster_path', 'tagline', 'production_companies', 'production_countries', 'spoken_languages'])
df_movie_score = pd.read_excel('../Dataset/Movie/movie_user_ratings.xlsx')
df_movie_score['userId'] = df_movie_score['userId'].astype('str')

df_movie = df_movie[df_movie['vote_average'] != 0]
df_movie_score = df_movie_score[df_movie_score['userId'].notnull()]

# Model Training
reader = Reader(rating_scale=(df_movie_score['rating'].min(), df_movie_score['rating'].max()))
data = Dataset.load_from_df(df_movie_score[['userId', 'tmdbId', 'rating']], reader)

# Define the SVD model
svd_model = SVD()

# Evaluate the model using cross-validation
cross_validate(svd_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the SVD model on the entire dataset
svd_trainset = data.build_full_trainset()
svd_model.fit(svd_trainset)

dump(svd_model, './API/Model/movie_svd_model.joblib')