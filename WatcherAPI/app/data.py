import pandas as pd
from joblib import load

def optimize_df(df):
    for col in df.select_dtypes(include=['float64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='float')
    for col in df.select_dtypes(include=['int64']).columns:
        df[col] = pd.to_numeric(df[col], downcast='integer')
    return df

# Anime Data
df_anime = pd.read_csv('./app/Dataset/Anime/anime.csv', usecols=['id', 'title', 'score', 'genres', 'type', 'studios', 'image_url'])
# df_anime = df_anime.drop(columns=['synopsis', 'episodes', 'status', 'producers', 'licensors', 'source', 'duration', 'rating', 'rank', 'popularity', 'favorites', 'scored_by', 'members', 'is_hentai'])
df_anime_score = pd.read_excel('./app/Dataset/Anime/anime_user_ratings.xlsx')
df_anime_score['user_id'] = df_anime_score['user_id'].astype('str')
anime_svd_model = load('./app/Model/anime_svd_model.joblib')

anime_scores = df_anime['score'][df_anime['score'] != -1]
anime_scores = anime_scores.astype('float32')
score_mean = round(anime_scores.mean() , 2)
df_anime['score'] = df_anime['score'].replace(-1, score_mean)
df_anime['score'] = df_anime['score'].astype('float32')
df_anime_score['rating'] = df_anime_score['rating'].astype('int32')
df_anime_score = df_anime_score[df_anime_score['user_id'].notnull()]

df_anime = optimize_df(df_anime)
df_anime_score = optimize_df(df_anime_score)

# Movie Data
df_movie = pd.read_csv('./app/Dataset/Movie/movie.csv', usecols=['id', 'title', 'vote_average', 'vote_count', 'release_date', 'poster_path', 'genres', 'production_companies', 'keywords'])
# df_movie = df_movie.drop(columns=['status', 'revenue', 'runtime', 'adult', 'backdrop_path', 'budget', 'homepage', 'imdb_id', 'original_language', 'original_title', 'overview', 'popularity', 'tagline', 'production_countries', 'spoken_languages'])
df_movie_score = pd.read_excel('./app/Dataset/Movie/movie_user_ratings.xlsx')
df_movie_score['userId'] = df_movie_score['userId'].astype('str')
movie_svd_model = load('./app/Model/movie_svd_model.joblib')

df_movie = df_movie[df_movie['vote_average'] != 0]
df_movie['vote_average'] = df_movie['vote_average'].astype('float32')
df_movie['vote_count'] = df_movie['vote_count'].astype('int32')
df_movie_score['rating'] = df_movie_score['rating'] * 2
df_movie_score['rating'] = df_movie_score['rating'].astype('int32')
df_movie_score = df_movie_score[df_movie_score['userId'].notnull()]

df_movie = optimize_df(df_movie)
df_movie_score = optimize_df(df_movie_score)

# TV Data
df_tv = pd.read_csv('./app/Dataset/TV/tv.csv', usecols=['id', 'name', 'vote_count', 'vote_average', 'first_air_date', 'in_production', 'poster_path', 'type', 'genres', 'production_companies'])
# df_tv = df_tv.drop(columns=['number_of_seasons', 'number_of_episodes', 'original_language', 'overview', 'adult', 'backdrop_path', 'last_air_date', 'homepage', 'original_name', 'popularity', 'status', 'tagline', 'created_by', 'languages', 'networks', 'origin_country', 'spoken_languages', 'production_countries', 'episode_run_time'])
df_tv_score = pd.read_csv('./app/Dataset/TV/tv_user_ratings.csv')
df_tv_score['userId'] = df_tv_score['userId'].astype('str')
tv_svd_model = load('./app/Model/tv_svd_model.joblib')

df_tv = df_tv[df_tv['vote_average'] != 0]
df_tv['vote_count'] = df_tv['vote_count'].astype('int32')
df_tv['vote_average'] = df_tv['vote_average'].astype('float32')
df_tv_score['rating'] = df_tv_score['rating'].astype('int32')
df_tv_score = df_tv_score[df_tv_score['userId'].notnull()]

df_tv = optimize_df(df_tv)
df_tv_score = optimize_df(df_tv_score)