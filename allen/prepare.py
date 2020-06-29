import pandas as pd
import numpy as np
from sklearn.experimental import enable_iterative_imputer  
from sklearn.impute import IterativeImputer
from sklearn.preprocessing import OneHotEncoder

#load Data

def correlation(dataset, threshold):
    col_corr = set() # Set of all the names of deleted columns
    corr_matrix = dataset.corr()
    for i in range(len(corr_matrix.columns)):
        for j in range(i):
            if (corr_matrix.iloc[i, j] >= threshold) and (corr_matrix.columns[j] not in col_corr):
                colname = corr_matrix.columns[i] # getting the name of column
                col_corr.add(colname)
                if colname in dataset.columns:
                    del dataset[colname] # deleting the column from the dataset

    return dataset

def one_hot_encoder(dataset):
    encoded_df = pd.DataFrame(index=dataset.index)
    features = dataset.columns.tolist()
    for i,feature in enumerate(features):
        ohe = OneHotEncoder(sparse=False, categories='auto')
        encoded_matrix = ohe.fit_transform(dataset[[feature]])
        encoded_dfi = pd.DataFrame(encoded_matrix, columns=ohe.categories_[0], index=dataset[feature].index)
        encoded_df = pd.concat([encoded_df,encoded_dfi], axis =1)
    return encoded_df

def prep_data(df):
    df=df.reset_index(drop = True)
    data_dict = pd.read_csv('data/WiDS Datathon 2020 Dictionary.csv')

    #Drop non_useful features
    identifier_features =data_dict[data_dict['Category']=='identifier']['Variable Name'].tolist() + ['icu_id']
    type__features=['hospital_admit_source','icu_admit_source', 'icu_stay_type','icu_type']
    redundant_features = ['readmission_status', 'apache_2_bodysystem']
    features_to_drop = identifier_features + type__features +redundant_features
    df = df.drop(columns=features_to_drop)

    #Handling Missing values
    cut_off_percentage = .3
    n_of_nulls = int(cut_off_percentage * df.shape[0])
    # drop features that have more than 70% of nulls
    df = df.dropna(axis=1, thresh = n_of_nulls)

    #Imputation of numeric variables
    #get numeric feature names 
    numeric_features = data_dict[data_dict['Data Type']=='numeric']['Variable Name'].tolist()+ ['bmi','apache_2_diagnosis','apache_3j_diagnosis']
    skewed_numeric_features = df.columns[df.columns.isin(numeric_features)]
    numeric_df = df[skewed_numeric_features]
    # Use iterative Imputer to fill in null with the information of other features
    imp = IterativeImputer(max_iter=3, verbose=0)
    imp.fit(numeric_df)
    imputed_df = imp.transform(numeric_df)
    imputed_df = pd.DataFrame(imputed_df, columns=numeric_df.columns)

    #Imputation of categorical variables
    categorical_features = data_dict[data_dict['Data Type']!='numeric']['Variable Name'].tolist()
    categorical_features =[feature for feature in categorical_features if feature not in ['bmi','apache_2_diagnosis','apache_3j_diagnosis']]
    skewed_categorical_features = df.columns[df.columns.isin(categorical_features)]
    categorical_df = df[skewed_categorical_features]
    # fill the null with the most occurred values
    # df.series.mode() returns a series. so [0] exact value of the series
    for feature in skewed_categorical_features:
        categorical_df[feature].fillna(categorical_df[feature].mode()[0],inplace=True)

    # concat two data frame together horizontally
    complet_df = pd.concat([imputed_df, categorical_df], axis = 1)

    # Create a new feature which is the gcs score
    complet_df['GCS'] = complet_df['gcs_eyes_apache'] + complet_df['gcs_motor_apache'] + complet_df['gcs_verbal_apache']
    complet_df = complet_df.drop(columns= ['gcs_eyes_apache', 'gcs_motor_apache', 'gcs_verbal_apache']) 
    
    # Reduce highly correlated numerical features
    reduced_dim_df = pd.concat([correlation(imputed_df.copy(), .9), categorical_df], axis = 1)
    reduced_dim_df
                                   
    # Label encoding gender
    complet_df['gender']= complet_df['gender'].apply(lambda x: 1 if x =='M' else 0)

    # One hot encoding ethnicity and appce 3j bodysystem
    dataset = complet_df[['ethnicity', 'apache_3j_bodysystem']]
    encoded_df = one_hot_encoder(dataset)

    # Concat encoded dataframe to the complete df
    # Drop the selected features
    complet_df = pd.concat([complet_df, encoded_df], axis=1).drop(columns =['ethnicity', 'apache_3j_bodysystem'])
    return complet_df










