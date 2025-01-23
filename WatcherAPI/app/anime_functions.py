import pandas as pd
import gc
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from app.data import df_anime

def get_user_item_anime_matrix(df_anime_score, df_real_time):
    # Combine the existing data with the real-time data
    df_combined = pd.concat([df_anime_score, df_real_time]).drop_duplicates(subset=['user_id', 'anime_id'], keep='last')
    
    # Create a pivot table with users as rows and animes as columns
    user_anime_matrix = df_combined.pivot(index='user_id', columns='anime_id', values='rating').fillna(0)
    user_click_matrix = df_combined.pivot(index='user_id', columns='anime_id', values='clickCount').fillna(0)
    user_favorite_matrix = df_combined.pivot(index='user_id', columns='anime_id', values='favoriteAt').fillna(0)

    del df_combined
    gc.collect()

    # Combine matrices
    user_anime_matrix = user_anime_matrix + user_click_matrix * 0.1 + user_favorite_matrix * 0.5
    
    # Convert to sparse matrix
    user_anime_sparse_matrix = csr_matrix(user_anime_matrix.values)
    
    return user_anime_matrix, user_anime_sparse_matrix

def get_anime_knn_model(user_anime_sparse_matrix):
    # Fit the NearestNeighbors model
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_anime_sparse_matrix)
    
    return knn

def get_anime_recommendations_by_user(user_id, knn, user_anime_matrix, user_anime_sparse_matrix, n):
    user_index = user_anime_matrix.index.get_loc(user_id)
    n_neighbors = min(n + 1, user_anime_sparse_matrix.shape[0])  # Ensure n_neighbors is not greater than the number of samples
    distances, indices = knn.kneighbors(user_anime_sparse_matrix[user_index], n_neighbors=n_neighbors)
    
    # Get the indices of the most similar users
    similar_users_indices = indices.flatten()[1:]  # Exclude the user itself
    similar_users_distances = distances.flatten()[1:]  # Exclude the user itself
    
    # Get the animes rated by the user
    user_rated_animes = user_anime_matrix.iloc[user_index]
    
    # Initialize a dictionary to store the weighted sum of ratings
    weighted_ratings = {}
    
    # Iterate over similar users
    for similar_user_index, distance in zip(similar_users_indices, similar_users_distances):
        similarity_score = 1 - distance
        similar_user_ratings = user_anime_matrix.iloc[similar_user_index]
        
        # Iterate over the animes rated by the similar user
        for anime_id, rating in similar_user_ratings.items():
            if user_rated_animes[anime_id] == 0:  # Only consider animes not rated by the user
                if anime_id not in weighted_ratings:
                    weighted_ratings[anime_id] = 0
                weighted_ratings[anime_id] += similarity_score * rating
    
    # Sort the animes based on the weighted sum of ratings
    sorted_animes = sorted(weighted_ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top n animes
    top_n_animes = sorted_animes[:n]
    return top_n_animes

def hybrid_anime_recommendations(user_id, svd, knn, user_anime_matrix, user_anime_sparse_matrix, n):
    svd_recommendations = []

    if svd is not None:
        # Get initial recommendations using the SVD model
        user_rated_animes = user_anime_matrix.loc[user_id]
        all_animes = user_anime_matrix.columns
        
        for anime_id in all_animes:
            if user_rated_animes[anime_id] == 0:  # Only consider animes not rated by the user
                svd_recommendations.append((anime_id, svd.predict(user_id, anime_id).est))

        # Normalize the SVD recommendation scores
        if svd_recommendations:
            max_svd_score = max(svd_recommendations, key=lambda x: x[1])[1]
            min_svd_score = min(svd_recommendations, key=lambda x: x[1])[1]
            if max_svd_score != min_svd_score:
                svd_recommendations = [(anime_id, (score - min_svd_score) / (max_svd_score - min_svd_score), score) for anime_id, score in svd_recommendations]
            else:
                svd_recommendations = [(anime_id, 0, score) for anime_id, score in svd_recommendations]
        
        # Sort the SVD recommendations
        svd_recommendations = sorted(svd_recommendations, key=lambda x: x[1], reverse=True)
        
        # Get the top n SVD recommendations
        top_svd_recommendations = svd_recommendations[:n]
    else:
        top_svd_recommendations = []
    
    # Refine the recommendations using user-based collaborative filtering
    user_based_recommendations = get_anime_recommendations_by_user(user_id, knn, user_anime_matrix, user_anime_sparse_matrix, n)

    # Normalize the user-based recommendation scores
    if user_based_recommendations:
        max_user_score = max(user_based_recommendations, key=lambda x: x[1])[1]
        min_user_score = min(user_based_recommendations, key=lambda x: x[1])[1]
        if max_user_score != min_user_score:
            user_based_recommendations = [(anime_id, (score - min_user_score) / (max_user_score - min_user_score), score) for anime_id, score in user_based_recommendations]
        else:
            user_based_recommendations = [(anime_id, 0, score) for anime_id, score in user_based_recommendations]
    
    # Combine both sets of recommendations and remove duplicates
    combined_recommendations = list({animeId: (score, original_score) for animeId, score, original_score in top_svd_recommendations + user_based_recommendations}.items())

    # Sort by recommendation score and then by tv score
    combined_recommendations = sorted(combined_recommendations, key=lambda x: (x[1][0], x[1][1]), reverse=True)

    final_recommendations = [anime_id for anime_id, _ in combined_recommendations[:n]]
    return final_recommendations

def get_anime_details_by_ids(anime_ids):
    animes = []
    for anime_id in anime_ids:
        anime = df_anime[df_anime['id'] == anime_id]
        animes.append(anime)
    result = pd.concat(animes)
    return result[['id', 'title', 'genres', 'score', 'image_url']]