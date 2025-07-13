from preprocess import *
from embeddings import *
from modelling.modelling import *
from modelling.data_model import ProcessedData
from data_loader import get_input_data
from Config import Config

import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer

#Set a seed so results are the same each time we run
seed_value = 0
random.seed(seed_value)
np.random.seed(seed_value)

def load_dataset():
    """
  This function loads the raw input data from CSV files and returns it as a DataFrame.
    """
    return get_input_data()

def clean_dataset(raw_df):
    """
This function removes duplicate entries and cleans any unwanted text or noise from the dataset.
    """
    dedup_df = de_duplication(raw_df)    #Remove repeated rows
    cleaned_df = noise_remover(dedup_df)    #Clean noisy text or formatting issues
    return cleaned_df

def vectorize_text(cleaned_df: pd.DataFrame):
    """
This function combines the summary and interaction columns into one text column,
    then converts this combined text into TF-IDF vectors for machine learning input.
    """
    cleaned_df['combined_text'] = (
        cleaned_df[Config.TICKET_SUMMARY].fillna('') + ' ' +
        cleaned_df[Config.INTERACTION_CONTENT].fillna('')
    ).str.lower()  #Convert all text to lowercase

    vectorizer = TfidfVectorizer()
    X = vectorizer.fit_transform(cleaned_df['combined_text'])

    return X, cleaned_df, vectorizer

def prepare_data(tfidf_matrix: np.ndarray, df: pd.DataFrame, vectorizer):
    """
This function wraps the TF-IDF features original data and vectorizer
    into a ProcessedData object which can be passed to the model.
    """
    return ProcessedData(tfidf_matrix, df, vectorizer=vectorizer)

def run_prediction(data_obj: ProcessedData, df: pd.DataFrame, group_name: str):
    """
This function sends the prepared data to the prediction pipeline,
    using the group name to label the output file.
    """
    model_predict(data_obj, df, group_name)

#This block runs when the script is executed directly
if __name__ == '__main__':
    dataset = load_dataset()      #Load CSV data
    dataset = clean_dataset(dataset)       #Clean and prepare the dataset

 #Make sure these columns are treated as strings (text)
  
    dataset[Config.INTERACTION_CONTENT] = dataset[Config.INTERACTION_CONTENT].astype(str)
    dataset[Config.TICKET_SUMMARY] = dataset[Config.TICKET_SUMMARY].astype(str)

#Group the dataset by a selected column (e.g... Mailbox)
    
    grouped_data = dataset.groupby(Config.GROUPED)

    for group_label, group_df in grouped_data:
        print(f"\nRunning model for group: {group_label}")

    #Convert the text to TF-IDF format
        tfidf_matrix, group_df, vectorizer = vectorize_text(group_df)

    #Create ProcessedData object with vectorized input
        data_obj = prepare_data(tfidf_matrix, group_df, vectorizer)

        #Run model and generate predictions
        run_prediction(data_obj, group_df, group_label)
