import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder


def split_data(df):
    X = df.drop(columns="hospital_death")
    y = df.hospital_death
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, train_size=0.8, random_seed=42, stratify=stratify_on
    )
    return X_train, X_test, y_train, y_test


def encode(df, l):
    """
    OneHot encodes the data in a column or columns
    """
    encoder = OneHotEncoder(sparse=False)
    encoder.fit(df[l])
    m = encoder.transform(df[l])
    col_name = encoder.get_feature_names(l)
    df = pd.concat([df, pd.DataFrame(m, columns=col_name, index=df.index)], axis=1)
    df = df.drop(columns=l)
    return df
