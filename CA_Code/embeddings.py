import numpy as np
import pandas as pd
from Config import *
import random

#Set random seed for reproducibility
seed = 0
random.seed(seed)
np.random.seed(seed)

def get_tfidf_embd(df: pd.DataFrame):
    """
    Create TF-IDF embeddings from text columns.
    
    It combines TICKET_SUMMARY and INTERACTION_CONTENT,
    then applies TF-IDF vectorization.
    """
    from sklearn.feature_extraction.text import TfidfVectorizer

    #Initialize TF-IDF vectorizer with reasonable settings
    tfidfconverter = TfidfVectorizer(
        max_features=2000,  #limit to top 2000 terms
        min_df=4,           #ignore words in fewer than 4 documents
        max_df=0.90         #ignore words in more than 90% of documents
    )

    #Combine the two text columns for better context
    data = df[Config.TICKET_SUMMARY] + ' ' + df[Config.INTERACTION_CONTENT]

    #Generate TF-IDF matrix and convert to NumPy array
    X = tfidfconverter.fit_transform(data).toarray()

    return X

def combine_embd(X1, X2):
    """
    Combine two embedding matrices horizontally (side-by-side).
    Useful when merging different types of embeddings.
    """
    return np.concatenate((X1, X2), axis=1)
