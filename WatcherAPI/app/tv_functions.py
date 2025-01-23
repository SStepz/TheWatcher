import pandas as pd
import gc
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from app.data import df_tv

def get_user_item_tv_matrix(df_tv_score, df_real_time):
    # Combine the existing data with the real-time data
    df_combined = pd.concat([df_tv_score, df_real_time]).drop_duplicates(subset=['userId', 'tmdbId'], keep='last')
    
    # Create a pivot table with users as rows and tvs as columns
    user_tv_matrix = df_combined.pivot(index='userId', columns='tmdbId', values='rating').fillna(0)
    user_click_matrix = df_combined.pivot(index='userId', columns='tmdbId', values='clickCount').fillna(0)
    user_favorite_matrix = df_combined.pivot(index='userId', columns='tmdbId', values='favoriteAt').fillna(0)

    del df_combined
    gc.collect()

    # Combine matrices
    user_tv_matrix = user_tv_matrix + user_click_matrix * 0.1 + user_favorite_matrix * 0.5
    
    # Convert to sparse matrix
    user_tv_sparse_matrix = csr_matrix(user_tv_matrix.values)
    
    return user_tv_matrix, user_tv_sparse_matrix

def get_tv_knn_model(user_tv_sparse_matrix):
    # Fit the NearestNeighbors model
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_tv_sparse_matrix)
    
    return knn

def get_tv_recommendations_by_user(user_id, knn, user_tv_matrix, user_tv_sparse_matrix, n):
    user_index = user_tv_matrix.index.get_loc(user_id)
    n_neighbors = min(n + 1, user_tv_sparse_matrix.shape[0])  # Ensure n_neighbors is not greater than the number of samples
    distances, indices = knn.kneighbors(user_tv_sparse_matrix[user_index], n_neighbors=n_neighbors)
    
    # Get the indices of the most similar users
    similar_users_indices = indices.flatten()[1:]  # Exclude the user itself
    similar_users_distances = distances.flatten()[1:]  # Exclude the user itself
    
    # Get the tvs rated by the user
    user_rated_tvs = user_tv_matrix.iloc[user_index]
    
    # Initialize a dictionary to store the weighted sum of ratings
    weighted_ratings = {}
    
    # Iterate over similar users
    for similar_user_index, distance in zip(similar_users_indices, similar_users_distances):
        similarity_score = 1 - distance
        similar_user_ratings = user_tv_matrix.iloc[similar_user_index]
        
        # Iterate over the tvs rated by the similar user
        for tmdbId, rating in similar_user_ratings.items():
            if user_rated_tvs[tmdbId] == 0:  # Only consider tvs not rated by the user
                if tmdbId not in weighted_ratings:
                    weighted_ratings[tmdbId] = 0
                weighted_ratings[tmdbId] += similarity_score * rating
    
    # Sort the tvs based on the weighted sum of ratings
    sorted_tvs = sorted(weighted_ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top n tvs
    top_n_tvs = sorted_tvs[:n]
    return top_n_tvs

def hybrid_tv_recommendations(user_id, svd, knn, user_tv_matrix, user_tv_sparse_matrix, n):
    svd_recommendations = []

    if svd is not None:
        # Get initial recommendations using the SVD model
        user_rated_tvs = user_tv_matrix.loc[user_id]
        all_tvs = user_tv_matrix.columns
        
        for tmdbId in all_tvs:
            if user_rated_tvs[tmdbId] == 0:  # Only consider tvs not rated by the user
                svd_recommendations.append((tmdbId, svd.predict(user_id, tmdbId).est))

        # Normalize the SVD recommendation scores
        if svd_recommendations:
            max_svd_score = max(svd_recommendations, key=lambda x: x[1])[1]
            min_svd_score = min(svd_recommendations, key=lambda x: x[1])[1]
            if max_svd_score != min_svd_score:
                svd_recommendations = [(tmdbId, (score - min_svd_score) / (max_svd_score - min_svd_score), score) for tmdbId, score in svd_recommendations]
            else:
                svd_recommendations = [(tmdbId, 0, score) for tmdbId, score in svd_recommendations]
        
        # Sort the SVD recommendations
        svd_recommendations = sorted(svd_recommendations, key=lambda x: x[1], reverse=True)
        
        # Get the top n SVD recommendations
        top_svd_recommendations = svd_recommendations[:n]
    else:
        top_svd_recommendations = []
    
    # Refine the recommendations using user-based collaborative filtering
    user_based_recommendations = get_tv_recommendations_by_user(user_id, knn, user_tv_matrix, user_tv_sparse_matrix, n)

    # Normalize the user-based recommendation scores
    if user_based_recommendations:
        max_user_score = max(user_based_recommendations, key=lambda x: x[1])[1]
        min_user_score = min(user_based_recommendations, key=lambda x: x[1])[1]
        if max_user_score != min_user_score:
            user_based_recommendations = [(tmdbId, (score - min_user_score) / (max_user_score - min_user_score), score) for tmdbId, score in user_based_recommendations]
        else:
            user_based_recommendations = [(tmdbId, 0, score) for tmdbId, score in user_based_recommendations]

    # Combine both sets of recommendations and remove duplicates
    combined_recommendations = list({tmdbId: (score, original_score) for tmdbId, score, original_score in top_svd_recommendations + user_based_recommendations}.items())

    # Sort by recommendation score and then by tv score
    combined_recommendations = sorted(combined_recommendations, key=lambda x: (x[1][0], x[1][1]), reverse=True)
    
    final_recommendations = [tmdbId for tmdbId, _ in combined_recommendations[:n]]
    return final_recommendations

def get_tv_details_by_ids(tv_ids):
    tvs = []
    for tv_id in tv_ids:
        tv = df_tv[df_tv['id'] == tv_id]
        tvs.append(tv)
    result = pd.concat(tvs)
    return result[['id', 'name', 'genres', 'vote_average', 'first_air_date', 'poster_path']]