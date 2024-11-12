import pandas as pd
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer
import re
import ast

def clean_text(df, filename=None):
    """
    This function tokenize, removes stop words, removes headers of text and applies stemming to the words to dataframe with column 'text'.
    """
    #remove header contained in brackets {}
    for text in df['text']:
        re.sub(r"\{([^}]*)\}", "", text)

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
        df.to_csv(filename)

    return df

def find_theme(text_df, theme_df):
    """
    Returns the normalized vector with  number of words in theme_df that each text in text_df contains
    """
    encoding = np.zeros(len(text_df))
    for i, text in enumerate(text_df['text']):
        for _, word in enumerate(theme_df['text'].values):
            if word in text:
                encoding[i] += 1
    max_val = np.max(encoding)
    encoding = encoding/max_val
    return encoding

def theme_encoding(theme_list):
    try:
        df = pd.read_csv("stemmed_movie_summaries.csv")
        df['text'] = df['text'].apply(lambda x: ast.literal_eval(x))
    except FileNotFoundError:
        df = pd.read_csv("data/original_data/plot_summaries.txt", delimiter = "\t", names=["id", "text"])
        df = clean_text(df, "stemmed_movie_summaries.csv")

    theme_df = pd.DataFrame({'text':theme_list})
    theme_df = clean_text(theme_df)
    theme_df['text'] = theme_df['text'].apply(lambda x: x[0])
    encoding = find_theme(df, theme_df)
    return encoding

#TEST
# death_synonyms = [ "killed", "murdered", "assassinated", "slain", "shot", "stabbed", "executed", "died", "perished", "succumbed", "bled", "decapitated", "strangled", "blown", "electrocuted", "drowned", "burned", "poisoned", "crushed", "impaled", "asphyxiated", "bludgeoned", "suffocated", "choked", "eviscerated", "slaughtered", "disintegrated", "torn", "gutted", "hanged", "gassed", "slashed", "ripped", "cut", "disemboweled", "blasted", "tortured", "beheaded", "smashed", "mauled", "knifed", "eaten", "struck", "dismembered", "skinned", "hacked", "pummeled", "pierced", "overpowered", "collapsed", "punched", "squashed", "sliced", "flayed", "stomped", "shattered", "flattened", "scalded", "severed", "hit", "pinned", "decayed", "blasted", "bludgeoned", "throttled", "attacked", "electrocuted", "incinerated" ]
# print(theme_encoding(death_synonyms))