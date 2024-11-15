from textblob import TextBlob
import pandas as pd

def sentiment_analysis(MOVIES):
    """
    Performs sentiment analysis on the 'plot' column of the MOVIES DataFrame and returns sentiment polarity scores.
    """

    return MOVIES['plot'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else None)