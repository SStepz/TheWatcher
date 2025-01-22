import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from joblib import dump

# TV Data
df_tv = pd.read_csv('../WatcherAPI/app/Dataset/TV/tv.csv')
df_tv = df_tv.drop(columns=['original_language', 'overview', 'adult', 'backdrop_path', 'first_air_date', 'last_air_date', 'homepage', 'original_name', 'poster_path', 'status', 'tagline', 'created_by', 'languages', 'networks', 'origin_country', 'spoken_languages', 'production_countries', 'episode_run_time'])
df_tv_score = pd.read_csv('../WatcherAPI/app/Dataset/TV/tv_user_ratings.csv')
df_tv_score['userId'] = df_tv_score['userId'].astype('str')

df_tv = df_tv[df_tv['vote_average'] != 0]
df_tv_score['rating'] = df_tv_score['rating'].astype('int64')
df_tv_score = df_tv_score[df_tv_score['userId'].notnull()]

# Model Training
reader = Reader(rating_scale=(df_tv_score['rating'].min(), df_tv_score['rating'].max()))
data = Dataset.load_from_df(df_tv_score[['userId', 'tmdbId', 'rating']], reader)

# Define the SVD model
svd_model = SVD()

# Evaluate the model using cross-validation
cross_validate(svd_model, data, measures=['RMSE', 'MAE'], cv=5, verbose=True)

# Train the SVD model on the entire dataset
svd_trainset = data.build_full_trainset()
svd_model.fit(svd_trainset)

dump(svd_model, '../WatcherAPI/app/Model/tv_svd_model.joblib')