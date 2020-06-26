import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder


# -----Dtype Tools-----
def convert_to_int_col(df):
    cols = ["age", "gcs_eyes_apache", "gcs_motor_apache", "gcs_verbal_apache"]
    for col in cols:
        df[col] = df[col].astype(int)
    return df


# ------Handling Nulls-----
def drop_cols_and_rows_by_threshold(
    df, prop_required_column=0.2, prop_required_row=0.4
):
    shape_before = df.shape
    threshold = int(round(prop_required_column * len(df.index), 0))
    df.dropna(axis=1, thresh=threshold, inplace=True)
    threshold = int(round(prop_required_row * len(df.columns), 0))
    df.dropna(axis=0, thresh=threshold, inplace=True)
    shape_aßfter = df.shape
    rows = shape_before[0] - shape_after[0]
    cols = shape_before[1] - shape_after[1]
    return df


def fill_with_mode(df):
    cols = [
        "age",
        "bmi",
        "ethnicity",
        "gender",
        "icu_admit_source",
        "hospital_admit_source",
        "apache_3j_bodysystem",
        "apache_2_bodysystem",
    ]
    for col in cols:
        if df[col].dtype == float:
            df[col].fillna(int(df[col].mode()), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
    return df


def fill_with_median(df):
    """
    Handles Nulls within the h1, d1 and apache features in a dataframe
    """
    d1_h1 = [col for col in df.columns if "d1" in col or "h1" in col]
    apache = [
        col
        for col in df.columns
        if "_apache" in col and "gcs" not in col and df[col].dtype != bool
    ]
    additional_cols = [
        "weight",
        "height",
        "apache_4a_icu_death_prob",
        "apache_4a_hospital_death_prob",
        "apache_2_diagnosis",
        "apache_3j_diagnosis",
    ]
    cols = d1_h1 + apache + additional_cols

    for col in cols:
        df[col].fillna(df[col].median(), inplace=True)
    return df


def fill_with_zero(df):
    """
    These data points are bools, so filling with False since that is the most common
    """
    cols = [
        "arf_apache",
        "intubated_apache",
        "cirrhosis",
        "aids",
        "diabetes_mellitus",
        "hepatic_failure",
        "immunosuppression",
        "leukemia",
        "lymphoma",
        "solid_tumor_with_metastasis",
        "gcs_unable_apache",
    ]

    for col in cols:
        df[col].fillna(0, inplace=True)
    return df


def fill_gcs(df):
    """
    The average GCS score for a human is 5, so specifying that for gcs related features
    """
    cols = ["gcs_eyes_apache", "gcs_motor_apache", "gcs_verbal_apache"]
    for col in cols:
        df[col].fillna(5, inplace=True)
    return df


def min_max_cols(df):
    min_max = []
    for col in df.columns:
        if "_min" in col and col.replace("_min", "_max") in df.columns:
            min_max.append(col)
    return min_max


def fix_min_max(df):
    """
    Fixes features where the min value is greater than the max value
    """

    min_max = min_max_cols(df)
    for col in min_max:
        vals = df[[col, col.replace("_min", "_max")]].values.copy()

        df[col] = np.nanmin(vals, axis=1)
        df[col.replace("_min", "_max")] = np.nanmax(vals, axis=1)
    return df


# Main Function
def get_raw_data():
    # Get the data into a pandas dataframe
    df = pd.read_csv("data/training_v2.csv")
    # df.set_index("patient_id", inplace=True)
    return df


def prepare_data_imp(df):
    """
    Takes in a Data Frame.
    Applies data transformation functions to handle null values. 
    Fixes features where the min values is greater than the max value.
    Returns the Data Frame
    """

    (
        df.pipe(fill_with_mode)
        .pipe(fill_with_median)
        .pipe(fill_with_zero)
        .pipe(fill_gcs)
        .pipe(convert_to_int_col)
    )
    return df


diagnosis_list =  [102.01, 501.05, 403.01, 104.01, 501.02, 212.01, 502.01, 203.01, 501.06,
 401.01, 106.01, 107.01, 206.01, 501.04, 501.01, 901.03, 211.09, 305.02, 410.01, 601.01,
 211.03, 101.01, 1401.01, 1404.01, 305.01, 212.02,
 207.01,
 409.02,
 109.09,
 202.01,
 204.01,
 201.01,
 306.01,
 407.01,
 704.01,
 1206.03,
 309.01,
 301.01,
 301.02,
 1502.02,
 1207.01,
 211.1,
 1501.01,
 402.02,
 211.02,
 802.01,
 109.1,
 703.03,
 1212.03,
 1410.01,
 601.06,
 111.01,
 103.01,
 311.01,
 702.01,
 108.01,
 211.12,
 109.03,
 211.08,
 701.02,
 601.05,
 211.04,
 402.01,
 1102.01,
 1904.01,
 208.01,
 601.04,
 405.01,
 1408.03,
 408.02,
 1302.02,
 105.02,
 1209.02,
 109.08,
 109.04,
 901.01,
 307.02,
 1204.02,
 1505.01,
 307.01,
 109.16,
 1205.01,
 1902.02,
 601.08,
 105.01,
 109.06,
 1406.01,
 1405.02,
 1601.01,
 110.01,
 1506.07,
 1202.07,
 603.01,
 213.01,
 310.01,
 109.14,
 801.04,
 312.05,
 1206.01,
 1210.01]

def diagnosis_agg(df):
    for i in range(0, df.shape[0]):
        if df.apache_3j_diagnosis[i] in diagnosis_list:
            df.apache_3j_diagnosis[i] = str(df.apache_3j_diagnosis[i])
        else:
            df.apache_3j_diagnosis[i] = 'other'
    return df

def prepare_data(df):
    """
    Takes in a Data Frame.
    Applies data transformation functions to handle null values. 
    Fixes features where the min values is greater than the max value.
    Returns the Data Frame
    """

    (
        # df.pipe(convert_to_int_col)
        df.pipe(fix_min_max)
        .pipe(diagnosis_agg)
    )
    return df