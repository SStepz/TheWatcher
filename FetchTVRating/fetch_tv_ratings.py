import requests
import pandas as pd
from openpyxl import load_workbook
import os

BASE_URL = 'https://api.themoviedb.org/3'

def fetch_tv_user_ratings(tv_id, page=1):
    url = f"{BASE_URL}/tv/{tv_id}/reviews?language=en-US&page={page}"
    headers = {
        "accept": "application/json",
        "Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9.eyJhdWQiOiI3NWZhNjNjNjU0ODllNzAyYTk5NjdmNTZjMTI2ZmNhMiIsIm5iZiI6MTcwNzM4OTUxMC41NDEsInN1YiI6IjY1YzRiMjQ2MGM0YzE2MDE4NTAzMTYxMyIsInNjb3BlcyI6WyJhcGlfcmVhZCJdLCJ2ZXJzaW9uIjoxfQ.VVeZBsFcQc7lltwE9WZGESfCRTo8S718QYGtyGbY0EQ"
    }
    response = requests.get(url, headers=headers)
    response.raise_for_status()
    return response.json()

def get_tv_user_ratings(tv_ids):
    user_ratings = []
    for tv_id in tv_ids:
        page = 1
        while True:
            data = fetch_tv_user_ratings(tv_id, page)
            if not data['results']:
                break
            for review in data['results']:
                if review['author_details']['rating'] is None or review['author_details']['rating'] == 0:
                    continue
                user_ratings.append({
                    'tmdbId': tv_id,
                    'userId': review['author_details']['username'],
                    'rating': review['author_details']['rating']
                })
            page += 1
    return user_ratings

# Example TV IDs (replace with actual TV IDs you want to fetch ratings for)
tv_ids = [93405, 215866, 236429, 125988, 125988, 219543, 112581, 94605, 244243, 79744, 194764, 95396, 157741, 91363, 219937, 239770]

# Load the CSV dataset
df = pd.read_csv('../AllData/TV/tv.csv')

# Filter TV IDs by vote_count > 0
filtered_df = df[df['vote_count'] > 4]
filtered_tv_ids = filtered_df['id'].unique()
test_ids = filtered_tv_ids[:100]

# Fetch user ratings
user_ratings = get_tv_user_ratings(test_ids)

# Convert to DataFrame
df = pd.DataFrame(user_ratings)

# Append to existing XLSX file
file_path = 'tv_user_ratings.csv'
if os.path.exists(file_path):
    df.to_csv(file_path, mode='a', header=False, index=False)
else:
    df.to_csv(file_path, index=False)