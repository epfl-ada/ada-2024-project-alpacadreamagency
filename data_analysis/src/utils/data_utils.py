import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import ast
from collections import Counter
import random
import torch
from sklearn.model_selection import train_test_split
from ..utils.settings import *

def load_movie_data():
    """
    Loads movie data from different sources based on the data version specified in SETTINGS.
    """

    if Settings.DATA_VERSION == 0:
        column_names = [
            "wikipedia_movie_ID", 
            "freebase_movie_ID", 
            "name", 
            "release_date", 
            "box_office_revenue", 
            "runtime", 
            "languages", 
            "countries", 
            "genres"
        ]
        MOVIES = pd.read_csv(f'{Settings.ORIGINAL_DATA_RUTE}movie.metadata.tsv', sep='\t', header=None, encoding='utf-8', na_values=['NA', ''], names=column_names)
        
    elif Settings.DATA_VERSION == 1: 
        MOVIES = pd.read_csv(f'{Settings.DATA_RUTE}TMDB_movie_dataset_v11.csv')
        MOVIES = MOVIES.drop(columns=["homepage", "poster_path", "backdrop_path"])
        
    elif Settings.DATA_VERSION == 2: 
        column_names = [
            "name", 
            "rating", 
            "genre", 
            "release_year", 
            "status", 
            "score", 
            "votes", 
            "director", 
            "writer", 
            "star",
            "country", 
            "budget", 
            "box_office_revenue", 
            "company",
            "runtime", 
        ]
        MOVIES = pd.read_csv(f'{Settings.DATA_RUTE}movies.csv', names=column_names)
        MOVIES = MOVIES.drop(columns=["director", "writer", "star"])
        MOVIES = MOVIES.drop(MOVIES.index[0])
        
    elif Settings.DATA_VERSION == 3:
        column_names = [
            "wikipedia_movie_ID", 
            "freebase_movie_ID", 
            "name", 
            "release_date", 
            "revenue", 
            "runtime", 
            "languages", 
            "countries", 
            "genres"
        ]
        MOVIES_ORIGINAL = pd.read_csv(f'{Settings.ORIGINAL_DATA_RUTE}movie.metadata.tsv', sep='\t', header=None, encoding='utf-8', na_values=['NA', ''], names=column_names)
        
        column_names = [
            'id_new', 
            'title', 
            'vote_average', 
            'vote_count', 
            'status_new', 
            'release_date_new',
            'revenue_new', 
            'runtime_new', 
            'adult', 
            'backdrop_path', 
            'budget', 
            'homepage',
            'imdb_id', 
            'original_language', 
            'original_title', 
            'overview',
            'popularity', 
            'poster_path', 
            'tagline', 
            'genres_new',
            'production_companies', 
            'production_countries', 
            'spoken_languages',
            'keywords'
        ]
        MOVIES_NEW = pd.read_csv(f'{Settings.DATA_RUTE}TMDB_movie_dataset_v11.csv', sep=',', header=None, encoding='utf-8', na_values=['NA', ''], names=column_names)
        
        
            
        MOVIES_ORIGINAL['name'] = MOVIES_ORIGINAL['name'].str.strip().str.lower()
        MOVIES_NEW['title'] = MOVIES_NEW['title'].str.strip().str.lower()
        
        MOVIES_ORIGINAL["release_year"] = pd.to_datetime(MOVIES_ORIGINAL['release_date'], errors='coerce').dt.year
        MOVIES_ORIGINAL["release_year"] = MOVIES_ORIGINAL["release_year"].fillna(0).astype(int)
        
        MOVIES_NEW["release_year_new"] = pd.to_datetime(MOVIES_NEW['release_date_new'], errors='coerce').dt.year
        MOVIES_NEW["release_year_new"] = MOVIES_NEW["release_year_new"].fillna(0).astype(int)
        
        MOVIES = pd.merge(MOVIES_ORIGINAL, MOVIES_NEW, left_on=['name', "release_year"], right_on=['title', 'release_year_new'], how='inner')
        MOVIES = MOVIES.drop(columns=[
            "id_new", "freebase_movie_ID", "title", "status_new","imdb_id",
            "original_language", "original_title","tagline", "genres_new", 'production_companies', 
            'production_countries', 'spoken_languages', 'keywords', "homepage", "poster_path", "backdrop_path"
        ])    
        
    elif Settings.DATA_VERSION== 4:
        column_names = [
            "wikipedia_movie_ID", 
            "freebase_movie_ID", 
            "name", 
            "release_date", 
            "revenue", 
            "runtime", 
            "languages", 
            "countries", 
            "genres"
        ]
        MOVIES_ORIGINAL = pd.read_csv(f'{Settings.ORIGINAL_DATA_RUTE}movie.metadata.tsv', sep='\t', header=None, encoding='utf-8', na_values=['NA', ''], names=column_names)
        
        column_names = [
            "names", 
            "date_x", 
            "score", 
            "genre", 
            "overview", 
            "crew", 
            "orig_title", 
            "status", 
            "orig_lang", 
            "budget_x", 
            "revenue", 
            "country"
        ]

        MOVIES_NEW = pd.read_csv(f'{Settings.DATA_RUTE}imdb_movies.csv', sep=',', header=None, encoding='utf-8', na_values=['NA', ''], names=column_names)
        
        
            
        MOVIES_ORIGINAL['name'] = MOVIES_ORIGINAL['name'].str.strip().str.lower()
        MOVIES_NEW['names'] = MOVIES_NEW['names'].str.strip().str.lower()
        
            
        MOVIES_ORIGINAL["release_year"] = pd.to_datetime(MOVIES_ORIGINAL['release_date'], errors='coerce').dt.year
        MOVIES_ORIGINAL["release_year"] = MOVIES_ORIGINAL["release_year"].fillna(0).astype(int)
        
        MOVIES_NEW["release_year_new"] = pd.to_datetime(MOVIES_NEW['date_x'], errors='coerce').dt.year
        MOVIES_NEW["release_year_new"] = MOVIES_NEW["release_year_new"].fillna(0).astype(int)
        
        MOVIES = pd.merge(MOVIES_ORIGINAL, MOVIES_NEW, left_on=['name'], right_on=['names'], how='left')
        # MOVIES = MOVIES.drop(columns=[
        #     "id_new", "title", "status_new", "revenue_new", "runtime_new", "imdb_id",
        #     "original_language", "original_title", "overview", "tagline", "genres_new", 'production_companies', 
        #     'production_countries', 'spoken_languages', 'keywords', "homepage", "poster_path", "backdrop_path"
        # ])  

    return MOVIES


def load_character_data():
    """
    Loads character data for movies from a TSV file based on the SETTINGS.
    """

    column_names = [
    "Wikipedia movie ID",
    "Freebase movie ID",
    "Movie release date",
    "Character name",
    "Actor date of birth",
    "Actor gender",
    "Actor height (in meters)",
    "Actor ethnicity (Freebase ID)",
    "Actor name",
    "Actor age at movie release",
    "Freebase character/actor map ID",
    "Freebase character ID",
    "Freebase actor ID"
    ]
    CHARACTER = pd.read_csv(f'{Settings.ORIGINAL_DATA_RUTE}character.metadata.tsv', sep='\t', header=None, encoding='utf-8', na_values=['NA', ''], names=column_names)
    
    return CHARACTER


def merge_plot_movies(MOVIES):
    """
    Reads a file containing plot summaries, then merges the plot data with the existing movie dataset (MOVIES)
    based on the ‘wikipedia_movie_ID’. 
    """

    column_names = [
    "wikipedia_movie_ID",
    "plot"
    ]

    PLOTS = pd.read_csv(f'{Settings.ORIGINAL_DATA_RUTE}plot_summaries.txt', sep='\t', header=None, encoding='utf-8', names=column_names)
    MOVIES = pd.merge(MOVIES, PLOTS, on="wikipedia_movie_ID", how="left")

    return MOVIES


def remove_duplicate_movies(MOVIES):
    """
    Removes duplicate movie entries in the MOVIES DataFrame, keeping the best values for each duplicated entry.
    """

    #get duplicated movies of MOVIES df
    dupes = MOVIES[MOVIES.duplicated(subset=['wikipedia_movie_ID'], keep=False)].sort_values(by="wikipedia_movie_ID", ascending=False)
    #keep highest value in columns: 
    for col in ['release_year','revenue', 'runtime', 'runtime_new', 'budget', 'popularity', 'revenue_new', 'release_date_new', 'adult' ]:
        dupes[col] = pd.to_numeric(dupes[col], errors='coerce').astype('float64')
        new = dupes.groupby('wikipedia_movie_ID')[col].transform('max')
        dupes[col] = new
    dupes['adult'] = dupes['adult'].astype('bool')

    #keep longest plot
    dupes['plot'] = dupes['plot'].astype('str')
    dupes['len_plot'] = dupes['plot'].apply(len)
    max_len_idx = dupes.groupby('wikipedia_movie_ID')['len_plot'].idxmax()
    dupes['plot'] = dupes['wikipedia_movie_ID'].map(dupes.loc[max_len_idx].set_index('wikipedia_movie_ID')['plot'])

    #keep vote avg and count of movie with highest vote count
    dupes['vote_count'] = pd.to_numeric(dupes['vote_count'], errors='coerce').astype('float64')
    max_len_idx = dupes.groupby('wikipedia_movie_ID')['vote_count'].idxmax()
    dupes['vote_average'] = dupes['wikipedia_movie_ID'].map(dupes.loc[max_len_idx].set_index('wikipedia_movie_ID')['vote_average'])
    dupes['vote_count'] = dupes.groupby('wikipedia_movie_ID')['vote_count'].transform('max')

    #list of the movies that had been duplicated, with the final parameters
    unduped = dupes.drop_duplicates(subset=['wikipedia_movie_ID'], keep='first')

    #remove the duplicates from MOVIES and insert the correct movies from unduped 
    MOVIES = MOVIES[~MOVIES.index.isin(dupes.index)]
    #MOVIES = pd.merge(MOVIES, unduped, on='wikipedia_movie_ID', how='outer')
    MOVIES = pd.concat([MOVIES, unduped], ignore_index=True)

    return MOVIES


def recover_from_new_db(MOVIES):
    """
    Recovers missing data in the runtime, revenue and plot columns of the MOVIES DataFrame by filling gaps
    with values from alternative columns.
 	"""
    MOVIES['runtime'] = MOVIES['runtime'].combine_first(MOVIES['runtime_new'])
    MOVIES['revenue'] = MOVIES['revenue'].combine_first(MOVIES['revenue_new'])
    MOVIES['plot'] = MOVIES['plot'].combine_first(MOVIES['overview'])

    return MOVIES

def clean_release_year(MOVIES):
    """
    Cleans the 'release_year' column in the MOVIES DataFrame by filtering out invalid or unrealistic years.
	"""

    # if "release_date" in MOVIES.columns:
    #     MOVIES["release_year"] = pd.to_datetime(MOVIES['release_date'], errors='coerce').dt.year
    if MOVIES["release_year"].dtype == "int64":
        MOVIES["release_year"] = MOVIES["release_year"].fillna(0).astype(int)
    MOVIES = MOVIES[(MOVIES["release_year"] >= Settings.FIRST_MOVIE_YEAR) & (MOVIES["release_year"] <= Settings.ACTUAL_YEAR) & (MOVIES["release_year"] != 0)]
    
    return MOVIES


def parse_features(MOVIES):
    """
    Parses and standardizes feature columns in the MOVIES DataFrame by converting specified columns to float type
    and encoding the 'adult' column as a binary integer.
    """
    MOVIES["budget"] = MOVIES["budget"].astype(float)
    MOVIES["popularity"] = MOVIES["popularity"].astype(float)
    MOVIES["revenue"] = MOVIES["revenue"].astype(float)
    MOVIES["runtime"] = MOVIES["runtime"].astype(float)
    MOVIES["vote_average"] = MOVIES["vote_average"].astype(float)
    MOVIES["vote_count"] = MOVIES["vote_count"].astype(float)
    #MOVIES = MOVIES[(MOVIES["runtime"] >= 40) & (MOVIES["runtime"] <= 200)]
    MOVIES["adult"] = MOVIES["adult"].apply(lambda x: 1 if x == "True" else 0)

    return MOVIES


def gather_subgenres(MOVIES, NEW_GENRE):
    """
    Maps movie genres to broader categories based on subgenre definitions and creates hot-encoded genre indicators.
    """

    def get_hot_genre(genre):
        """
        Creates hot-encoded genre indicators for genres.
        """
        new_genre = []
        genre_hot = []
        
        movie_genre = set(genre)
        
        for _, row in NEW_GENRE.iterrows():
            sub_genres = set(row["subgenres"])
            if bool(movie_genre & sub_genres):
                new_genre.append(row["categories"])
                genre_hot.append(1)
            else: genre_hot.append(0)
        
        return new_genre, genre_hot
    MOVIES["original_genres"] = MOVIES["genres"].apply(lambda x: ast.literal_eval(x).values())
    MOVIES[["new_genres", "genre_hot"]] = MOVIES["original_genres"].apply(lambda x: pd.Series(get_hot_genre(x)))

    return MOVIES


def count_characters(MOVIES, CHARACTER):
    """
    Adds a column to the MOVIES DataFrame representing the total number of characters per movie.
    """

    total_character_counts = CHARACTER.groupby('Wikipedia movie ID').size().reset_index(name='Character Count')

    # We drop 'Character Count' column in MOVIES if it exists to avoid conflict during the merge
    if 'Character Count' in MOVIES.columns:
        MOVIES = MOVIES.drop(columns=['Character Count'])

    #total_character_counts = total_character_counts.rename(columns={'Wikipedia movie ID': 'movie_id_temp'})

    # Merge this data into MOVIES based on Wikipedia movie ID
    #MOVIES = MOVIES.merge(total_character_counts, left_on='wikipedia_movie_ID', right_on='movie_id_temp', how='left')
    MOVIES = MOVIES.merge(total_character_counts, left_on='wikipedia_movie_ID', right_on='Wikipedia movie ID', how='left')
    MOVIES['Character Count'] = MOVIES['Character Count'].fillna(0).astype(int)
    MOVIES['Character Count'] = MOVIES['Character Count'].astype(int)

    if 'Wikipedia movie ID' in MOVIES.columns:
        MOVIES = MOVIES.drop(columns=['Wikipedia movie ID'])

    return MOVIES


def count_genders(CHARACTER):
    """
    Counts the occurrences of male, female, and unknown/unspecified actors for each movie.
    """


    CHARACTER['Actor gender filled'] = CHARACTER['Actor gender'].fillna('N/A')

    # We use pivot_table to count occurrences of each gender per Wikipedia movie ID
    actor_counts = CHARACTER.pivot_table(
        index='Wikipedia movie ID',
        columns='Actor gender filled',
        aggfunc='size',
        fill_value=0
    ).reset_index()

    # Renaming some columns
    actor_counts = actor_counts.rename(columns={
        'M': 'Male actor count',
        'F': 'Female actor count',
        'N/A': 'N/A actor count'
    })

    return actor_counts


def merge_genders_movies(MOVIES, actor_counts):
    """ 
    Merges actor-gender counts into the MOVIES dataset.
    """ 


    if 'Male actor count' in MOVIES.columns:
        MOVIES = MOVIES.drop(columns=['Male actor count'])
    if 'Female actor count' in MOVIES.columns:
        MOVIES = MOVIES.drop(columns=['Female actor count'])
    if 'N/A actor count' in MOVIES.columns:
        MOVIES = MOVIES.drop(columns=['N/A actor count'])


    # Merge gender counts data back into the MOVIES DataFrame
    MOVIES = MOVIES.merge(actor_counts, left_on='wikipedia_movie_ID', right_on='Wikipedia movie ID', how='left')

    MOVIES['Male actor count'] = MOVIES['Male actor count'].fillna(0).astype(int)
    MOVIES['Female actor count'] = MOVIES['Female actor count'].fillna(0).astype(int)
    MOVIES['N/A actor count'] = MOVIES['N/A actor count'].fillna(0).astype(int)

    if 'Wikipedia movie ID' in MOVIES.columns:
        MOVIES = MOVIES.drop(columns=['Wikipedia movie ID'])

    # Replace NaN values with 0 
    #MOVIES['Number of Female'] = MOVIES['Number of Female'].fillna(0).astype(int)
    #MOVIES['Number of Male'] = MOVIES['Number of Male'].fillna(0).astype(int)
    #MOVIES['Number with no gender'] = MOVIES['Character Count']- MOVIES['Number of Female'] - MOVIES['Number of Male']  

    return MOVIES, ["Male actor count", "Female actor count", "N/A actor count"]


def add_actor_per_age(MOVIES, CHARACTER):
    """
    Processes the CHARACTER DataFrame to categorize actors into age ranges based on their age at the movie release. 
    It then counts the number of actors in each age range for every movie and merges this data into the MOVIES DataFrame. 
    """

    def categorize_age(age):
        if age < 20:
            return '0-20'
        elif 20 <= age < 30:
            return '20-30'
        elif 30 <= age < 40:
            return '30-40'
        elif 40 <= age < 60:
            return '40-60'
        else:
            return '60+'

    column_names = ['Actors 0-20', 'Actors 20-30', 'Actors 30-40', 'Actors 40-60', 'Actors 60+']

    CHARACTER['age_range'] = CHARACTER['Actor age at movie release'].apply(categorize_age)

    age_counts = CHARACTER.groupby(['Wikipedia movie ID', 'age_range']).size().unstack(fill_value=0)

    age_counts.columns = column_names

    if column_names[0] not in MOVIES.columns:
        MOVIES = MOVIES.merge(age_counts, left_on='wikipedia_movie_ID', right_index=True, how='left')

    MOVIES[column_names] = MOVIES[column_names].fillna(0).astype(int)

    return MOVIES, column_names