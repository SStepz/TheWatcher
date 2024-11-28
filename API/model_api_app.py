from flask import Flask, request, jsonify
from joblib import load
from pathlib import Path
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

app = Flask(__name__)

# Load the model and data
svd_model = load(Path('Model/svd_model.joblib'))
df_score = pd.read_csv('Dataset/Anime/users-scores.csv')
user_anime_matrix = df_score.pivot(index='user_id', columns='anime_id', values='rating').fillna(0)
user_similarity = cosine_similarity(user_anime_matrix)
user_similarity_df = pd.DataFrame(user_similarity, index=user_anime_matrix.index, columns=user_anime_matrix.index)

# Function to get recommendations
def get_recommendations_by_user(user_id, user_similarity_df, user_anime_matrix, n=10):
    similar_users = user_similarity_df[user_id].sort_values(ascending=False)
    user_rated_animes = user_anime_matrix.loc[user_id]
    weighted_ratings = {}
    for similar_user_id, similarity_score in similar_users.iteritems():
        if similar_user_id == user_id:
            continue
        similar_user_ratings = user_anime_matrix.loc[similar_user_id]
        for anime_id, rating in similar_user_ratings.iteritems():
            if user_rated_animes[anime_id] == 0:
                if anime_id not in weighted_ratings:
                    weighted_ratings[anime_id] = 0
                weighted_ratings[anime_id] += similarity_score * rating
    sorted_animes = sorted(weighted_ratings.items(), key=lambda x: x[1], reverse=True)
    top_n_animes = [anime_id for anime_id, _ in sorted_animes[:n]]
    return top_n_animes

@app.route('/recommendations', methods=['GET'])
def recommendations():
    user_id = int(request.args.get('user_id'))
    recommendations = get_recommendations_by_user(user_id, user_similarity_df, user_anime_matrix)
    return jsonify(recommendations)

if __name__ == '__main__':
    app.run(debug=True)