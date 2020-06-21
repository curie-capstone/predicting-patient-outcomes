import pandas as pd
import os.path
from sklearn.model_selection import train_test_split
from prepare import prep_data

def split():
    df =  pd.read_csv('data/training_v2.csv')
    train, test = train_test_split(df, train_size = .80, random_state = 42, stratify = df.hospital_death)
    train = prep_data(train)
    train.to_csv('data/train.csv')
    test = prep_data(test)
    test.to_csv('data/test.csv')
    return train, test

def acquire_train_test_data():
    if os.path.exists('data/train.csv') & os.path.exists('data/test.csv'):
        train = pd.read_csv('data/train.csv',  index_col=0)
        test = pd.read_csv('data/test.csv',  index_col=0)
    else:
        train,test = split()
    return train,test