def encode_cols(df):
    encoder = LabelEncoder()
    cols = ['ethnicity', 'gender']
    for col in cols:
        df[col] = encoder.fit_transform(df[col])
    
