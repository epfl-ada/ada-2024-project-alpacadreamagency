from textblob import TextBlob
import pandas as pd

def sentiment_analysis(MOVIES):
    return MOVIES['plot'].apply(lambda x: TextBlob(str(x)).sentiment.polarity if pd.notnull(x) else None)