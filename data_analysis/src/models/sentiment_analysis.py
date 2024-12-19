from textblob import TextBlob
import pandas as pd

def sentiment_analysis(MOVIES):
    """
        Get the sentiment analysis of each plot: How possitive or negative is each move based on its plot.
    """
    return MOVIES['plot'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else None)