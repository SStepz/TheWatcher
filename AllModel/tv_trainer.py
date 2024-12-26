import numpy as np
import pandas as pd
from surprise import SVD, Dataset, Reader
from surprise.model_selection import cross_validate
from joblib import dump

# TV Data
df_tv = pd.read_csv('../Dataset/TV/tv.csv')
df_tv = df_tv.drop(columns=['original_language', 'overview', 'adult', 'backdrop_path', 'first_air_date', 'last_air_date', 'homepage', 'original_name', 'poster_path', 'status', 'tagline', 'created_by', 'languages', 'networks', 'origin_country', 'spoken_languages', 'production_companies', 'production_countries', 'episode_run_time'])
# df_tv_score = pd.read_excel('../Dataset/TV/tv_user_ratings.xlsx')

df_tv = df_tv[df_tv['vote_average'] != 0]