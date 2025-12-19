import pandas as pd 
import pickle as pkl 

def split_train_test(dataframe):
    split_idx = int(len(dataframe) * 0.8)
    df_train_split = dataframe[:split_idx]  # First 80%
    df_valid = dataframe[split_idx:]         # Last 20%
    df_train = df_train_split
    return df_train, df_valid

def prepare_features(dataframe):
    y_train = dataframe['sold']
    y_valid = dataframe['sold']
    X_train = dataframe.drop('sold', axis=1)
    X_valid = dataframe.drop('sold', axis=1)
    return X_train, X_valid, y_train, y_valid