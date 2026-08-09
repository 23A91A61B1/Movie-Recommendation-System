import streamlit as st
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

movies = pd.read_pickle("movies.pkl")

cv = CountVectorizer(max_features=5000, stop_words="english")
vectors = cv.fit_transform(movies["tags"]).toarray()

similarity = cosine_similarity(vectors)

def recommend(movie):
    movie_index = movies[movies["title"] == movie].index[0]
    distances = similarity[movie_index]

    movie_list = sorted(
        list(enumerate(distances)),
        reverse=True,
        key=lambda x: x[1]
    )[1:6]

    return [movies.iloc[i[0]].title for i in movie_list]

st.title("Movie Recommendation System")

movie = st.selectbox("Select a movie", movies["title"].values)

if st.button("Recommend"):
    recommendations = recommend(movie)

    st.subheader("Recommended Movies")
    for m in recommendations:
        st.write(m)
