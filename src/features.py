def add_gcs(df):
    df['gcs'] = (df.gcs_eyes_apache 
                 + df.gcs_motor_apache 
                 + df.gcs_verbal_apache)
    df.drop(columns=['gcs_eyes_apache', 
                     'gcs_motor_apache',
                     'gcs_verbal_apache'],
           inplace=True)
    return df