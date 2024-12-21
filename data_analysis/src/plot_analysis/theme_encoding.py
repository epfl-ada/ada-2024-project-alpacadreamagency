import pandas as pd
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
import ast
import os
import json

nltk.download("stopwords")


def create_theme_df():
    """
    Create and returns a dataframe for the theme words in the themes dictionary.
    """

    with open(r"src\plot_analysis\encoded_themes.json", 'r') as json_file:
        themes = json.load(json_file)

    df_themes = pd.Series(themes, name='subwords')
    df_themes.index.name = "themes"

    return df_themes, list(themes.keys())



def clean_text(df, filename=None):
    """
    This function tokenize, removes stop words and applies stemming to the words to dataframe.
    Saves the dataframe to filename if given.
    """
    
    #remove punctuation
    for p in string.punctuation:
        df['text'] = df['text'].str.replace(p, "")
    df['text'] = df['text'].str.split(" ")

    stop_words = set(stopwords.words("english"))

    stemmer = PorterStemmer()

    def remove_stopwords_and_stem(text):
        """
        This function removes stopwords and stems the words
        It also removes empty strings created by the removal of punctuation
        """
        return [stemmer.stem(word.lower()) for word in text if ((word.lower() not in stop_words) and (len(word)>1) and (word.lower()))]


    df['text'] = df['text'].apply(remove_stopwords_and_stem)

    if filename:
        df.to_csv(filename) #saves file as csv if name given

    return df

def find_theme(text_df, theme_df):
    """
    Returns list of number of words in theme_df that each text in text_df contains
    """
    encoding = np.zeros(len(text_df))
    for i, text in enumerate(text_df['text']):
        for _, word in enumerate(theme_df['text'].values):
            if word in text:
                encoding[i] += 1

    return encoding

def theme_encoding(plots):
    """
    Returns a matrix with normalized (along the themes) encoding for each synopsis
    in the plots dataframe for each topic.
    """
    file_path = r"src\plot_analysis\stemmed_movie_summaries.csv"
    # If file path already exist read file, else create a new file with the cleaned plots
    if os.path.exists(file_path):
        df = pd.read_csv(file_path)
        df['text'] = df['text'].apply(lambda x: ast.literal_eval(x))
    else:
        df = pd.DataFrame({"text": plots.values})
        df = clean_text(df, "stemmed_movie_summaries.csv")

    themes, topic_names = create_theme_df()
    matrix = np.zeros((len(plots), len(themes)))

    for idx, theme_list in enumerate(themes):
        theme_df = pd.DataFrame({'text':theme_list})
        theme_df = clean_text(theme_df)
        theme_df['text'] = theme_df['text'].apply(lambda x: x[0])
        encoding = find_theme(df, theme_df)
        matrix[:, idx] = encoding #each row coresponds to one plot, and each column is one theme

    # Normalize along the themes (rows)
    for i in range(matrix.shape[0]):
        norm = np.sum(matrix[i, :])
        if norm != 0:
            matrix[i, :] = matrix[i, :]/norm
        
    return matrix, topic_names
#TEST
# death_synonyms = [ "killed", "murdered", "assassinated", "slain", "shot", "stabbed", "executed", "died", "perished", "succumbed", "bled", "decapitated", "strangled", "blown", "electrocuted", "drowned", "burned", "poisoned", "crushed", "impaled", "asphyxiated", "bludgeoned", "suffocated", "choked", "eviscerated", "slaughtered", "disintegrated", "torn", "gutted", "hanged", "gassed", "slashed", "ripped", "cut", "disemboweled", "blasted", "tortured", "beheaded", "smashed", "mauled", "knifed", "eaten", "struck", "dismembered", "skinned", "hacked", "pummeled", "pierced", "overpowered", "collapsed", "punched", "squashed", "sliced", "flayed", "stomped", "shattered", "flattened", "scalded", "severed", "hit", "pinned", "decayed", "blasted", "bludgeoned", "throttled", "attacked", "electrocuted", "incinerated" ]
# print(theme_encoding(death_synonyms))