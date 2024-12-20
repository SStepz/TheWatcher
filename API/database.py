from pymongo import MongoClient
from decouple import config
import pandas as pd

mongouser = config("MONGO_USERNAME")
mongopass = config("MONGO_PASSWORD")

def fetch_real_time_data(data_type):
    # Connect to the MongoDB database
    client = MongoClient(f"mongodb+srv://{mongouser}:{mongopass}@cluster0.ls3onag.mongodb.net/next-auth-prisma?retryWrites=true&w=majority&appName=Cluster0")
    db = client['next-auth-prisma']
    collection = db['Media']

    # Fetch the latest user ratings
    if data_type == 'anime':
        cursor = collection.find({'mediaType': 'anime', 'point': {'$gte': 1}}, {'_id': 0, 'userId': 1, 'mediaId': 1, 'point': 1})
        df_real_time = pd.DataFrame(list(cursor))
        df_real_time.rename(columns={'userId': 'user_id', 'mediaId': 'anime_id', 'point': 'rating'}, inplace=True)
        df_real_time['user_id'] = df_real_time['user_id'].astype('str')
        df_real_time['anime_id'] = df_real_time['anime_id'].astype('int64')
    elif data_type == 'movie':
        cursor = collection.find({'mediaType': 'movie', 'point': {'$gte': 1}}, {'_id': 0, 'userId': 1, 'mediaId': 1, 'point': 1})
        df_real_time = pd.DataFrame(list(cursor))
        df_real_time.rename(columns={'userId': 'userId', 'mediaId': 'tmdbId', 'point': 'rating'}, inplace=True)
        df_real_time['userId'] = df_real_time['userId'].astype('str')
        df_real_time['tmdbId'] = df_real_time['tmdbId'].astype('int64')
    elif data_type == 'tv':
        cursor = collection.find({'mediaType': 'serie', 'point': {'$gte': 1}}, {'_id': 0, 'userId': 1, 'mediaId': 1, 'point': 1})
    
    # Close the connection
    client.close()
    
    return df_real_time