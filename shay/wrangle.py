import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

def get_data():
    # Get the data into a pandas dataframe
    df =  pd.read_csv('data/training_v2.csv')
    print(f'rows: {df.shape[0]}, columns: {df.shape[1]}')
    return df

def drop_cols_and_rows_by_threshold(df, prop_required_column = .2, prop_required_row = .4):
    shape_before = df.shape
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    shape_after = df.shape
    rows = shape_before[0] - shape_after[0]
    cols = shape_before[1] - shape_after[1]
    print(f'\t * Number of patients dropped: {rows} \n \t * Number of features dropped:{cols}')
    return df
    
def fill_with_mode(df):
    cols = ['age', 'bmi',
            'hospital_admit_source']
    for col in cols:
        df[col].fillna(df[col].mode(), inplace=True)
        print('    * ' + col)
    return df

# Main Function
def get_training_data():
    '''
    Reads in the data as a pandas dataframe. Handles null values.
    ''' 
    print('---Acquiring the Data---')
    df = get_data()
    
    # Apply data transformations through helper functions
    print('\n')
    print('---Handling Missing Values---')
    print('Filling nulls with mode for the following features:')
    df.pipe(fill_with_mode)
    print('Dropping features and patients with too many missing values')
    df.pipe(drop_cols_and_rows_by_threshold)
    return df
    