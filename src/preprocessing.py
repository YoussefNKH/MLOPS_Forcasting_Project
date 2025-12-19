import pandas as pd 
import pickle as pkl 

def split_train_test(dataframe):
    split_idx = int(len(dataframe) * 0.8)
    df_train_split = dataframe[:split_idx]  
    df_valid = dataframe[split_idx:]        
    df_train = df_train_split
    return df_train, df_valid

def prepare_features(df_train, df_valid):
    y_train = df_train['sold']
    y_valid = df_valid['sold']
    X_train = df_train.drop('sold', axis=1)
    X_valid = df_valid.drop('sold', axis=1)
    return X_train, X_valid, y_train, y_valid