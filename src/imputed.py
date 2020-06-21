from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer

import pandas as pd


def get_raw_data():
    # Get the data into a pandas dataframe
    df = pd.read_csv("data/training_v2.csv")
#     print(f"rows: {df.shape[0]}, columns: {df.shape[1]}")
    df.set_index("patient_id", inplace=True)
    return df


def main():
    df = get_raw_data()
    data_dict = pd.read_csv("data/WiDS Datathon 2020 Dictionary.csv")

    identifier_features = data_dict[data_dict["Category"] == "identifier"][
        "Variable Name"
    ].tolist() + ["icu_id"]
    type__features = [
        "hospital_admit_source",
        "icu_admit_source",
        "icu_stay_type",
        "icu_type",
    ]
    redundant_features = ['readmission_status', 'apache_2_bodysystem']
    features_to_drop = identifier_features + type__features + redundant_features

    # keep features that have less than 70% of nulls
    cut_off_percentage = 0.3
    n_of_nulls = int(cut_off_percentage * df.shape[0])
    df = df.dropna(axis=1, thresh=n_of_nulls)

    numeric_features = data_dict[data_dict["Data Type"] == "numeric"][
        "Variable Name"
    ].tolist() + ["bmi", "apache_2_diagnosis", "apache_3j_diagnosis"]

    skewed_numeric_features = df.columns[df.columns.isin(numeric_features)]
    numeric_df = df[skewed_numeric_features]

    imp = IterativeImputer(max_iter=3, verbose=0)
    imp.fit(numeric_df)
    imputed_df = imp.transform(numeric_df)
    imputed_df = pd.DataFrame(imputed_df, columns=numeric_df.columns)

    categorical_features = data_dict[data_dict["Data Type"] != "numeric"][
        "Variable Name"
    ].tolist()

    # remove ['bmi','apache_2_diagnosis','apache_3j_diagnosis'] non_categorical features
    categorical_features = [
        feature
        for feature in categorical_features
        if feature not in ["bmi", "apache_2_diagnosis", "apache_3j_diagnosis"]
    ]

    skewed_categorical_features = df.columns[df.columns.isin(categorical_features)]

    categorical_df = df[skewed_categorical_features]

    # fill the null with the most occurred values


    # df.series.mode() returns a series. so [0] exact value of the series
    for feature in skewed_categorical_features:
        categorical_df[feature].fillna(categorical_df[feature].mode()[0], inplace=True)

    complete_df = pd.concat([imputed_df, categorical_df], axis=1)

    return complete_df
