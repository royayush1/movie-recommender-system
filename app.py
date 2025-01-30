import os
import streamlit as st
import pandas as pd
from joblib import load
from surprise import Reader, Dataset
from annoy import AnnoyIndex

#Load trained SVD Model and cache result to avoid unnecessary reloading
@st.cache_resource
def load_model(model_path):
    return load(model_path)

#Load Movies Dataset and cache result to avoid rereading
@st.cache_data
def load_movies(movies_path):
    return pd.read_csv(movies_path)

#Collaborative Filtering. Returning the movies it recommends and their respective ratings
def get_top_n_cf_recommendations(algo, user_id, movies_df, ratings_df, n=10):
    """
    Generate top N collaborative model based movie recommendations for a user

    Parameters:
    - algo: Trained Surprise Algorithm
    - user_id: ID of the user
    - movies_df: Dataframe containing the movies data
    - ratings_df: Dataframe containing the ratings data
    - n: Number of recommendations to return

    Returns:
    - List of tuples (movieId, predicted_rating)

    """
    all_movie_ids = movies_df['movieId'].unique()
    
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()

    unrated_movie_ids = [movie for movie in all_movie_ids if movie not in rated_movie_ids]

    predictions = [algo.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    
    predictions.sort(key = lambda x: x.est, reverse=True)

    i = 0
    counter = 0
    genres_detected = {}
    top_n = []
    while counter < 10:
        current_id = predictions[i].iid
        current_movie_genre = movies_df[movies_df['movieId'] == current_id]['genres'].astype(str).iloc[0]
        if genres_detected.get(current_movie_genre, -1) == -1:
            genres_detected[current_movie_genre] = 0
        if genres_detected[current_movie_genre] < 2:
            top_n.append(predictions[i])
            genres_detected[current_movie_genre] += 1
            counter += 1
        i+=1       
    final_top10_pred = []
    for pred in top_n:
        final_top10_pred.append((pred.iid, pred.est))
            
    return final_top10_pred



# Get Top 10 Content based recommendations based on annoy index (cosine similarity matrix is impractical)
def get_content_based_recommendations(movie_id, movies_df, index, top_n=10, num_movies=67000):
    """
    Generate top N content based movie recommendations for a user

    Parameters:
    - movie_id: ID of the movie
    - movies_df: Dataframe containing the movies data
    - index: Annoy Index
    - top_n: Number of recommendations to return
    - num_movies - Number of movies in Annoy index

    Returns:
    - Dataframe of recommended movies

    """
    idx = movies_df.index[movies_df['movieId'] == movie_id].tolist()[0]
    similar_indices = []
    if idx < num_movies:
        similar_indices = index.get_nns_by_item(idx, top_n+1)
    else:
        similar_indices = [idx]
    similar_movies = movies_df.iloc[similar_indices][['movieId', 'title', 'genres']]
    return similar_movies


# Generate Hybrid Recommendations using both our collaborative filtering model and content based methods
def hybrid_recommendations(algo, user_id, movies_df, ratings_df, index, top_n_cf=10, top_n_cb=10, num_movies=67000):
    """
    Generate top N hybrid based movie recommendations for a user. Combines and balances suggestions from above collaborative filtering model and content based model

    Parameters:
    - algo: Trained Surprise Algorithm
    - user_id: ID of the user
    - movies_df: Dataframe containing the movies data
    - ratings_df: Dataframe containing the ratings data
    - index: Annoy Index
    - top_n_cf: Number of recommendations to return by collaborative filtering model
    - top_n_cb: Number of recommendations to return by content based model
    - num_movies - Number of movies in Annoy index

    Returns:
    - List of top 10 recommended hybrid suggestions for movies

    """
    cf_recs = get_top_n_cf_recommendations(algo, user_id, movies_df, ratings_df, n = top_n_cf)

    combined_recs = {}

    for i in range(len(cf_recs)):
        (movie_id, predicted_rating) = cf_recs[i]
        similar_movies = get_content_based_recommendations(movie_id, movies_df, index, top_n=top_n_cb, num_movies=num_movies)
        for _, row in similar_movies.iterrows():
            sim_movie_id = row['movieId']
            title = row['title']
            #Strike a balance between suggestions from collaborative model and content model by assigning different scores
            if sim_movie_id not in combined_recs:
                if sim_movie_id == movie_id:
                    combined_recs[sim_movie_id] = {'title':title, 'count': 2}
                else:
                    combined_recs[sim_movie_id] = {'title':title, 'count': 1}
                if similar_movies.shape[0] == 1 and i < 10:
                    combined_recs[sim_movie_id] = {'title':title, 'count': 4}  
            elif sim_movie_id == movie_id:
                combined_recs[sim_movie_id]['count'] = combined_recs[sim_movie_id]['count'] + 1 * combined_recs[sim_movie_id]['count']
            else:
                combined_recs[sim_movie_id]['count'] = combined_recs[sim_movie_id]['count'] + 2 * combined_recs[sim_movie_id]['count']
            

    recs_list = [{'movieId': mid, 'title': info['title'], 'score': info['count']} for mid, info in combined_recs.items()]

    recs_df = pd.DataFrame(recs_list)
    recs_df = recs_df.sort_values(by='score', ascending=False).head(10)
    recs_list = recs_df.values.tolist()

    top10_list = [movies_df[movies_df['movieId'] == int(movie[0])]['title'].values[0] for movie in recs_list]
    top10_list = top10_list[:10]

    return top10_list

def main():
    st.title("Movie Recommendation System")
    st.write("Enter your UserID to get personalized Movie Recommendations: ")

    #Check if files exist
    if not os.path.exists('svd_model.joblib'):
        st.error("SVD Model not found. Please ensure svd_model.joblib is present in the same directory")
    if not os.path.exists('annoy_index.ann'):
        st.error("Annoy Index not found. Please ensure annoy_index.ann is present in the same directory")

    #Load SVD Model
    model = load_model("svd_model.joblib")   
    st.success("SVD Model Loaded Successfully")

    #Load movie and ratings data
    movies = load_movies("ml-32m/movies.csv")
    ratings = pd.read_csv("ml-32m/ratings.csv")

    #Load Annoy Index
    num_features = 42994
    index = AnnoyIndex(num_features, 'angular')
    index.load('annoy_index.ann')
    st.success("Annoy Index Loaded Successfully")

    #Input user id into app 
    user_id = st.number_input("User ID", min_value=1, max_value=240003, step=1, value=1)

    if st.button("Get Recommendations"):
        with st.spinner("Generating Recommendations...."):
            hybrid_recs = hybrid_recommendations(model, user_id, movies, ratings, index, 10, 10, 67000)
            st.success("Here are your top 10 movie recommendations: ")
            for idx, title in enumerate(hybrid_recs, start=1):
                st.write(f"{idx}. {title}")

if __name__ == "__main__":
    main()
                
            
    
    