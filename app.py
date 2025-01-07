import streamlit as st
import pandas as pd
from joblib import load
from surprise import Reader, Dataset

#Load trained SVD Model and cache result to avoid unnecessary reloading
@st.cache(allow_output_mutation=True)
def load_model(model_path):
    return load(model_path)

#Load Movies Dataset and cache result to avoid rereading
@st.cache
def load_movies(movies_path):
    return pd.read_csv(movies_path)

def get_top_n_recommendations(algo, user_id, movies_df, ratings_df, n=20):
    # Get all movie IDs
    all_movie_ids = movies_df['movieId'].unique()
    
    # Get movies already rated by the user
    rated_movie_ids = ratings_df[ratings_df['userId'] == user_id]['movieId'].unique()
    
    # Movies not yet rated
    unrated_movie_ids = [movie for movie in all_movie_ids if movie not in rated_movie_ids]
    
    # Predict ratings for all unrated movies
    predictions = [algo.predict(user_id, movie_id) for movie_id in unrated_movie_ids]
    
    # Sort predictions by estimated rating in descending order
    predictions.sort(key=lambda x: x.est, reverse=True)
    
    # Get top N recommendations
    top_n = predictions[:n]
    
    # Get movie titles
    top_n_movies = [(movies_df[movies_df['movieId'] == int(pred.iid)]['title'].values[0], pred.est) for pred in top_n]
    
    return top_n_movies

def main():
    st.title("Movie Recommendation System")
    st.write("Enter your UserID to get personalized Movie Recommendations: ")

    model = load_model("svd_model.joblib")
    movies = load_movies("ml-32m/movies.csv")
    ratings = pd.read_csv("ml-32m/ratings.csv")

    #Input user id into app 
    user_id = st.number_input("User ID", min_value=1, max_value=240003, step=1, value=1)

    if st.button("Get Recommendations"):
        with st.spinner("Generating Recommendations...."):
            top_recommendations = get_top_n_recommendations(model, user_id, movies, ratings, n=20)
            st.success("Here are your top 20 movie recommendations: ")
            for idx, (title, rating) in enumerate(top_recommendations, start=1):
                st.write(f"{idx}. {title} (Predicted Rating: {rating:.2f})")

if __name__ == "__main__":
    main()
                
            
    
    