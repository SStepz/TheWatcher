import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
from app.data import df_anime, df_movie

# Function to compute cosine similarity for a single anime
def compute_cosine_similarity_for_anime(matrix, idx):
    anime_vector = matrix[idx]
    similarity_scores = cosine_similarity(anime_vector, matrix).flatten()
    return similarity_scores

def get_user_item_anime_matrix(df_anime_score, df_real_time):
    # Combine the existing data with the real-time data
    df_combined = pd.concat([df_anime_score, df_real_time]).drop_duplicates(subset=['user_id', 'anime_id'], keep='last')
    
    # Create a pivot table with users as rows and animes as columns
    user_anime_matrix = df_combined.pivot(index='user_id', columns='anime_id', values='rating').fillna(0)
    
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
    distances, indices = knn.kneighbors(user_anime_sparse_matrix[user_index], n_neighbors=n+1)
    
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
    # Get initial recommendations using the SVD model
    user_rated_animes = user_anime_matrix.loc[user_id]
    all_animes = user_anime_matrix.columns
    svd_recommendations = []
    
    for anime_id in all_animes:
        if user_rated_animes[anime_id] == 0:  # Only consider animes not rated by the user
            svd_recommendations.append((anime_id, svd.predict(user_id, anime_id).est))

    # Normalize the SVD recommendation scores
    max_svd_score = max(svd_recommendations, key=lambda x: x[1])[1]
    min_svd_score = min(svd_recommendations, key=lambda x: x[1])[1]
    svd_recommendations = [(anime_id, (score - min_svd_score) / (max_svd_score - min_svd_score), score) for anime_id, score in svd_recommendations]
    
    # Sort the SVD recommendations
    svd_recommendations = sorted(svd_recommendations, key=lambda x: x[1], reverse=True)
    
    # Get the top n SVD recommendations
    top_svd_recommendations = svd_recommendations[:n]
    
    # Refine the recommendations using user-based collaborative filtering
    user_based_recommendations = get_anime_recommendations_by_user(user_id, knn, user_anime_matrix, user_anime_sparse_matrix, n)

    # Normalize the user-based recommendation scores
    max_user_score = max(user_based_recommendations, key=lambda x: x[1])[1]
    min_user_score = min(user_based_recommendations, key=lambda x: x[1])[1]
    user_based_recommendations = [(anime_id, (score - min_user_score) / (max_user_score - min_user_score), score) for anime_id, score in user_based_recommendations]
    
    # Combine both sets of recommendations and remove duplicates
    combined_recommendations = list(set(top_svd_recommendations + user_based_recommendations))

    # Sort by recommendation score and then by anime score
    combined_recommendations = sorted(combined_recommendations, key=lambda x: (x[1], x[2]), reverse=True)

    final_recommendations = [anime_id for anime_id, _, _ in combined_recommendations[:n]]
    return final_recommendations

def get_anime_details_by_ids(anime_ids):
    animes = []
    for anime_id in anime_ids:
        anime = df_anime[df_anime['id'] == anime_id]
        animes.append(anime)
    result = pd.concat(animes)
    return result[['id','title', 'genres', 'score']]



def get_user_item_movie_matrix(df_movie_score, df_real_time):
    # Combine the existing data with the real-time data
    df_combined = pd.concat([df_movie_score, df_real_time]).drop_duplicates(subset=['userId', 'tmdbId'], keep='last')
    
    # Create a pivot table with users as rows and movies as columns
    user_movie_matrix = df_combined.pivot(index='userId', columns='tmdbId', values='rating').fillna(0)
    
    # Convert to sparse matrix
    user_movie_sparse_matrix = csr_matrix(user_movie_matrix.values)
    
    return user_movie_matrix, user_movie_sparse_matrix

def get_movie_knn_model(user_movie_sparse_matrix):
    # Fit the NearestNeighbors model
    knn = NearestNeighbors(metric='cosine', algorithm='brute')
    knn.fit(user_movie_sparse_matrix)
    
    return knn

def get_movie_recommendations_by_user(user_id, knn, user_movie_matrix, user_movie_sparse_matrix, n):
    user_index = user_movie_matrix.index.get_loc(user_id)
    distances, indices = knn.kneighbors(user_movie_sparse_matrix[user_index], n_neighbors=n+1)
    
    # Get the indices of the most similar users
    similar_users_indices = indices.flatten()[1:]  # Exclude the user itself
    similar_users_distances = distances.flatten()[1:]  # Exclude the user itself
    
    # Get the movies rated by the user
    user_rated_movies = user_movie_matrix.iloc[user_index]
    
    # Initialize a dictionary to store the weighted sum of ratings
    weighted_ratings = {}
    
    # Iterate over similar users
    for similar_user_index, distance in zip(similar_users_indices, similar_users_distances):
        similarity_score = 1 - distance
        similar_user_ratings = user_movie_matrix.iloc[similar_user_index]
        
        # Iterate over the movies rated by the similar user
        for tmdbId, rating in similar_user_ratings.items():
            if user_rated_movies[tmdbId] == 0:  # Only consider movies not rated by the user
                if tmdbId not in weighted_ratings:
                    weighted_ratings[tmdbId] = 0
                weighted_ratings[tmdbId] += similarity_score * rating
    
    # Sort the movies based on the weighted sum of ratings
    sorted_movies = sorted(weighted_ratings.items(), key=lambda x: x[1], reverse=True)
    
    # Get the top n movies
    top_n_movies = sorted_movies[:n]
    return top_n_movies

def hybrid_movie_recommendations(user_id, svd, knn, user_movie_matrix, user_movie_sparse_matrix, n):
    # Get initial recommendations using the SVD model
    user_rated_movies = user_movie_matrix.loc[user_id]
    all_movies = user_movie_matrix.columns
    svd_recommendations = []
    
    for tmdbId in all_movies:
        if user_rated_movies[tmdbId] == 0:  # Only consider movies not rated by the user
            svd_recommendations.append((tmdbId, svd.predict(user_id, tmdbId).est))

    # Normalize the SVD recommendation scores
    max_svd_score = max(svd_recommendations, key=lambda x: x[1])[1]
    min_svd_score = min(svd_recommendations, key=lambda x: x[1])[1]
    svd_recommendations = [(tmdbId, (score - min_svd_score) / (max_svd_score - min_svd_score), score) for tmdbId, score in svd_recommendations]
    
    # Sort the SVD recommendations
    svd_recommendations = sorted(svd_recommendations, key=lambda x: x[1], reverse=True)
    
    # Get the top n SVD recommendations
    top_svd_recommendations = svd_recommendations[:n]
    
    # Refine the recommendations using user-based collaborative filtering
    user_based_recommendations = get_movie_recommendations_by_user(user_id, knn, user_movie_matrix, user_movie_sparse_matrix, n)

    # Normalize the user-based recommendation scores
    max_user_score = max(user_based_recommendations, key=lambda x: x[1])[1]
    min_user_score = min(user_based_recommendations, key=lambda x: x[1])[1]
    user_based_recommendations = [(tmdbId, (score - min_user_score) / (max_user_score - min_user_score), score) for tmdbId, score in user_based_recommendations]

    # Combine both sets of recommendations and remove duplicates
    combined_recommendations = list(set(top_svd_recommendations + user_based_recommendations))

    # Sort by recommendation score and then by movie score
    combined_recommendations = sorted(combined_recommendations, key=lambda x: (x[1], x[2]), reverse=True)
    
    final_recommendations = [tmdbId for tmdbId, _, _ in combined_recommendations[:n]]
    return final_recommendations

def get_movie_details_by_ids(movie_ids):
    movies = []
    for movie_id in movie_ids:
        movie = df_movie[df_movie['id'] == movie_id]
        movies.append(movie)
    result = pd.concat(movies)
    return result[['id','title', 'genres', 'vote_average']]