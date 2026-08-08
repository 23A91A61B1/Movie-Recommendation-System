import numpy as np
import pandas as pd

import zipfile
import os
import pandas as pd # Ensure pandas is imported

csv_file_movies = '/content/tmdb_5000_movies.csv'
csv_file_credits = '/content/tmdb_5000_credits.csv'

# Define potential source files (actual zips or misnamed csvs)
source_zip_movies = '/content/tmdb_5000_movies.csv.zip'
source_zip_credits = '/content/tmdb_5000_credits.csv.zip'

# Helper function to extract or rename if it's a misnamed CSV
def ensure_csv_file(target_csv_path, source_file_with_zip_ext):
    if not os.path.exists(target_csv_path):
        if os.path.exists(source_file_with_zip_ext):
            try:
                # Try to unzip if it's a real zip
                with zipfile.ZipFile(source_file_with_zip_ext, 'r') as zip_ref:
                    zip_ref.extractall(os.path.dirname(target_csv_path))
                print(f"Successfully unzipped {source_file_with_zip_ext}")
                return True
            except zipfile.BadZipFile:
                # If it's not a zip, assume it's a misnamed CSV and rename
                print(f"Warning: '{source_file_with_zip_ext}' is not a valid zip file. Attempting to treat it as a CSV by renaming.")
                os.rename(source_file_with_zip_ext, target_csv_path)
                return True
            except Exception as e:
                print(f"Error processing {source_file_with_zip_ext}: {e}")
                return False
        else:
            print(f"Error: Neither '{target_csv_path}' nor '{source_file_with_zip_ext}' found.")
            return False
    else:
        print(f"'{target_csv_path}' already exists.")
        return True

# Ensure movies.csv is available
ensure_csv_file(csv_file_movies, source_zip_movies)

# Ensure credits.csv is available
ensure_csv_file(csv_file_credits, source_zip_credits)

movies = pd.read_csv(csv_file_movies)
credits = pd.read_csv(csv_file_credits)

movies.head()

credits.head()

movies = movies.merge(credits,on='title')

movies.head(1)

#genres
#id
#title
#overview
#cast
#crew

movies = movies[['movie_id' , 'title' , 'overview' , 'genres' , 'keywords' , 'cast' , 'crew']]

movies.head()

movies.isnull().sum()

movies.dropna(inplace=True)

movies.duplicated().sum()

movies.iloc[0].genres

import ast

def convert(obj):
  L = []
  for i in ast.literal_eval(obj):
    L.append(i['name'])
  return L

movies['genres'] = movies['genres'].apply(convert)

movies.head()

movies['keywords'] = movies['keywords'].apply(convert)

movies.head()

def convert3(obj):
  L = []
  counter = 0
  for i in ast.literal_eval(obj):
    if counter != 3:
      L.append(i['name'])
      counter+=1
    else:
      break
  return L

movies['cast'] = movies['cast'].apply(convert3)

movies.head()

def fetch_directors(obj):
  L = []
  for i in ast.literal_eval(obj):
    if i['job'] == 'Director':
      L.append(i['name'])
      break
  return L

movies['crew'] = movies['crew'].apply(fetch_directors)

movies.head()

movies['overview'][0]

movies['overview'] = movies['overview'].apply(lambda x:x.split())

movies.head()

movies['genres'] = movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords'] = movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast'] = movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew'] = movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies.head()

movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']

movies.head()

new_df = movies[['movie_id' , 'title' , 'tags']]

new_df['tags'] = new_df['tags'].apply(lambda x:" ".join(x))

import nltk

from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()

def stem(text):
  y = []

  for i in text.split():
    y.append(ps.stem(i))
  return " ".join(y)

new_df['tags'] = new_df['tags'].apply(stem)

new_df['tags'][0]

new_df['tags'] = new_df['tags'].apply(lambda x:x.lower())

new_df.head()

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features=5000,stop_words='english')

vector = cv.fit_transform(new_df['tags']).toarray()

vector

cv.get_feature_names_out()

from sklearn.metrics.pairwise import cosine_similarity

similarity = cosine_similarity(vector)

sorted(list(enumerate(similarity[0])),reverse=True,key=lambda x:x[1])[1:6]

def recommend(movie):
  movie_index = new_df[new_df['title'] == movie].index[0]
  distances = similarity[movie_index]
  movies_list = sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]

  for i in movies_list:
    print(new_df.iloc[i[0]].title)

recommend('Avatar')

import pickle

pickle.dump(new_df, open('movies.pkl', 'wb'))
pickle.dump(similarity, open('similarity.pkl', 'wb'))

