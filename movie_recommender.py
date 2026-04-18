import streamlit as st
import pandas as pd
import ast
import requests
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

API_KEY = "18707ad3e1e52fc9da0f2ed3d10ebf91"

st.set_page_config(page_title="Movie Recommender System", layout="wide")

# 🎨 Netflix UI
st.markdown("""
<style>
body { background-color: #141414; }
.title {
    font-size:50px;
    color:#E50914;
    text-align:center;
    font-weight:bold;
}
.subtitle {
    text-align:center;
    color:white;
    margin-bottom:30px;
}
.movie-title {
    color:white;
    font-size:16px;
    text-align:center;
}
</style>
""", unsafe_allow_html=True)

st.markdown('<div class="title">🎬 Movie Recommender System</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Discover movies like Netflix 🍿</div>', unsafe_allow_html=True)

# ---------------- LOAD ----------------
movies = pd.read_csv('tmdb_5000_movies.csv')
credits = pd.read_csv('credits_small.csv')

movies = movies.merge(credits, on='title')

movies = movies[['movie_id','title','overview','genres','keywords','cast','crew','vote_average','release_date']]
movies.dropna(inplace=True)

# ---------------- PROCESS ----------------
def convert(text):
    return [i['name'] for i in ast.literal_eval(text)]

movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x: convert(x)[:3])

def fetch_director(text):
    return [i['name'] for i in ast.literal_eval(text) if i['job'] == 'Director']

movies['crew'] = movies['crew'].apply(fetch_director)
movies['overview'] = movies['overview'].apply(lambda x: x.split())

for col in ['genres','keywords','cast','crew']:
    movies[col] = movies[col].apply(lambda x: [i.replace(" ","") for i in x])

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

new_df = movies[['movie_id','title','tags','vote_average','release_date']]
new_df['tags'] = new_df['tags'].apply(lambda x: " ".join(x).lower())

# ---------------- MODEL ----------------
tfidf = TfidfVectorizer(max_features=5000, stop_words='english')
vectors = tfidf.fit_transform(new_df['tags']).toarray()
similarity = cosine_similarity(vectors)

# ---------------- API FUNCTIONS ----------------
def fetch_poster(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}?api_key={API_KEY}"
        data = requests.get(url, timeout=5).json()
        if data.get('poster_path'):
            return "https://image.tmdb.org/t/p/w500" + data['poster_path']
        return "https://via.placeholder.com/500x750?text=No+Image"
    except:
        return "https://via.placeholder.com/500x750?text=Error"

def fetch_trailer(movie_id):
    try:
        url = f"https://api.themoviedb.org/3/movie/{movie_id}/videos?api_key={API_KEY}"
        data = requests.get(url).json()
        for video in data['results']:
            if video['type'] == 'Trailer':
                return f"https://www.youtube.com/watch?v={video['key']}"
        return None
    except:
        return None

# ---------------- RECOMMEND ----------------
def recommend(movie):
    idx = new_df[new_df['title'] == movie].index[0]
    distances = similarity[idx]

    movies_list = sorted(list(enumerate(distances)), reverse=True, key=lambda x:x[1])[1:6]

    names, posters, ratings, years, trailers = [], [], [], [], []

    for i in movies_list:
        row = new_df.iloc[i[0]]
        movie_id = row.movie_id

        names.append(row.title)
        posters.append(fetch_poster(movie_id))
        ratings.append(round(row.vote_average,1))
        years.append(row.release_date[:4])
        trailers.append(fetch_trailer(movie_id))

    return names, posters, ratings, years, trailers

# ---------------- UI ----------------
movie_list = new_df['title'].values
selected_movie = st.selectbox("🔍 Search a movie", movie_list)

if st.button("🔥 Recommend"):
    names, posters, ratings, years, trailers = recommend(selected_movie)

    st.markdown("## 🍿 Recommended for You")

    cols = st.columns(5)

    for i in range(5):
        with cols[i]:
            st.image(posters[i])
            st.markdown(f"<div class='movie-title'><b>{names[i]}</b></div>", unsafe_allow_html=True)
            st.markdown(f"<div class='movie-title'>⭐ {ratings[i]} | 📅 {years[i]}</div>", unsafe_allow_html=True)

            if trailers[i]:
                st.markdown(f"[▶ Watch Trailer]({trailers[i]})")
