import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Acquiring the Data
def get_data():
    # Get the data into a pandas dataframe
    df =  pd.read_csv('data/training_v2.csv')
    print(f'rows: {df.shape[0]}, columns: {df.shape[1]}')
    return df


# Dtype Tools
def convert_to_int_col(col):
    col = col.astype(int)
    

def drop_cols_and_rows_by_threshold(df, prop_required_column = .2, prop_required_row = .4):
    shape_before = df.shape
    threshold = int(round(prop_required_column*len(df.index),0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row*len(df.columns),0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    shape_aßfter = df.shape
    rows = shape_before[0] - shape_after[0]
    cols = shape_before[1] - shape_after[1]
    print(f'\t * Number of patients dropped: {rows} \n \t * Number of features dropped:{cols}')
    return df

    
def fill_with_mode(df):
    cols = ['age', 'bmi',
            'ethnicity']
    for col in cols:
        if df[col].dtype == float:
            df[col].fillna(int(df[col].mode()), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
        print('\t * ' + col)
    return df


def drop_rows(df):
    print('\t * Dropping patients with no recorded gender')
    num_patients = df.gender.isna().sum()
    df.dropna(subset=['gender'], inplace=True)
    print(f'\t\t - Number of patients dropped: {num_patients}')
    return df


def fill_with_zero(df):
    cols = ['arf_apache', 'intubated_apache',
            'cirrhosis', 'aids', 'diabetes_mellitus',
            'hepatic_failure', 'immunosuppression',
            'leukemia', 'lymphoma', 
            'solid_tumor_with_metastasis',
            'hospital_death', 'ethnicity',
            'gender', 'age', 'bmi']
    
    for col in cols:
        df[col].fillna(0, inplace=True)
        print('\t * ' + col)
    return df

def min_max_cols(df):
    min_max = []
    for col in df.columns:
        if '_min' in col and col.replace('_min', '_max') in df.columns:
            min_max.append(col)
    return min_max
        
def fix_min_max(df):
    min_max = min_max_cols(df)
    for col in min_max:
        vals = df[[col, col.replace('_min', '_max')]].values.copy()

        df[col] = np.nanmin(vals, axis=1)
        df[col.replace('_min', '_max')] = np.nanmax(vals, axis=1)

# Main Function
def get_training_data():
    '''
    Reads in the data as a pandas dataframe. Handles null values.
    ''' 
    print('---Acquiring the Data---')
    df = get_data()
    
    print('\n')
    
    print('---Handling Missing Values---')
    print('Filling nulls with mode for the following features:')
    df.pipe(fill_with_mode)
    print('Handling nulls within rows')
    df.pipe(drop_rows)
    print('Filling nulls with 0 (aka False) for the following columns')
    df.pipe(fill_with_zero)
    fix_min_max(df)
    return df
    