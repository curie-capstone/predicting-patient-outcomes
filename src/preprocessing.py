import pandas as pd
from sklearn.preprocessing import OneHotEncoder

def encode(df, l):
    '''
    OneHot encodes the data in a column or columns
    '''
    encoder = OneHotEncoder(sparse = False)
    encoder.fit(df[l])
    m = encoder.transform(df[l])
    col_name= encoder.get_feature_names(l)
    df = pd.concat([df, pd.DataFrame(m, columns = col_name,index = df.index)], axis =1)
    df = df.drop(columns = l)
    return df