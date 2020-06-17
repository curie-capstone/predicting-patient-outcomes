import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


def drop_nulls(df):
    df.dropna(inplace=True)
    
    return df
    
def encode_cols(df):
    encoder = LabelEncoder()
    
    cols = ['ethnicity', 'gender']
    for col in cols:
        df[col] = encoder.fit_transform(df[col])
        
    return df

# Main Function
def get_training_data():
    '''
    Reads in the data as a pandas dataframe
    ''' 
    df =  pd.read_csv('data/training_v2.csv')
  
    df = df[['bmi','age',
             'gender','ethnicity',
             'hospital_death']]
    
    df = drop_nulls(df)
    df = encode_cols(df)
    
    return df
    