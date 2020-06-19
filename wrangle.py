import features
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Acquiring the Data
def get_raw_data():
    # Get the data into a pandas dataframe
    df = pd.read_csv("data/training_v2.csv")
    print(f"rows: {df.shape[0]}, columns: {df.shape[1]}")
    df.set_index("patient_id", inplace=True)
    return df


def select_columns_to_use(df):
    cols = [
        "bmi",
        "age",
        "gender",
        "ethnicity",
        "solid_tumor_with_metastasis",
        "lymphoma",
        "leukemia",
        "immunosuppression",
        "hepatic_failure",
        "diabetes_mellitus",
        "aids",
        "cirrhosis",
        "intubated_apache",
        "hospital_death",
        "arf_apache",
        "elective_surgery",
        "icu_admit_source",
        "icu_id",
        "icu_stay_type",
        "gcs_eyes_apache",
        "gcs_motor_apache",
        "gcs_verbal_apache",
        "d1_diasbp_invasive_max",
        "d1_diasbp_invasive_min",
        "d1_diasbp_max",
        "d1_diasbp_min",
        "d1_diasbp_noninvasive_max",
        "d1_diasbp_noninvasive_min",
        "d1_heartrate_max",
        "d1_heartrate_min",
        "d1_mbp_invasive_max",
        "d1_mbp_invasive_min",
        "d1_mbp_max",
        "d1_mbp_min",
        "d1_mbp_noninvasive_max",
        "d1_mbp_noninvasive_min",
        "d1_resprate_max",
        "d1_resprate_min",
        "d1_spo2_max",
        "d1_spo2_min",
        "d1_sysbp_invasive_max",
        "d1_sysbp_invasive_min",
        "d1_sysbp_max",
        "d1_sysbp_min",
        "d1_sysbp_noninvasive_max",
        "d1_sysbp_noninvasive_min",
        "d1_temp_max",
        "d1_temp_min",
        "d1_albumin_max",
        "d1_albumin_min",
        "d1_bilirubin_max",
        "d1_bilirubin_min",
        "d1_bun_max",
        "d1_bun_min",
        "d1_calcium_max",
        "d1_calcium_min",
        "d1_creatinine_max",
        "d1_creatinine_min",
        "d1_glucose_max",
        "d1_glucose_min",
        "d1_hco3_max",
        "d1_hco3_min",
        "d1_hemaglobin_max",
        "d1_hemaglobin_min",
        "d1_hematocrit_max",
        "d1_hematocrit_min",
        "d1_inr_max",
        "d1_inr_min",
        "d1_lactate_max",
        "d1_lactate_min",
        "d1_platelets_max",
        "d1_platelets_min",
        "d1_potassium_max",
        "d1_potassium_min",
        "d1_sodium_max",
        "d1_sodium_min",
        "d1_wbc_max",
        "d1_wbc_min",
        "d1_arterial_pco2_max",
        "d1_arterial_pco2_min",
        "d1_arterial_ph_max",
        "d1_arterial_ph_min",
        "d1_arterial_po2_max",
        "d1_arterial_po2_min",
        "d1_pao2fio2ratio_max",
        "d1_pao2fio2ratio_min",
        "h1_diasbp_invasive_max",
        "h1_diasbp_invasive_min",
        "h1_diasbp_max",
        "h1_diasbp_min",
        "h1_diasbp_noninvasive_max",
        "h1_diasbp_noninvasive_min",
        "h1_heartrate_max",
        "h1_heartrate_min",
        "h1_mbp_invasive_max",
        "h1_mbp_invasive_min",
        "h1_mbp_max",
        "h1_mbp_min",
        "h1_mbp_noninvasive_max",
        "h1_mbp_noninvasive_min",
        "h1_resprate_max",
        "h1_resprate_min",
        "h1_spo2_max",
        "h1_spo2_min",
        "h1_sysbp_invasive_max",
        "h1_sysbp_invasive_min",
        "h1_sysbp_max",
        "h1_sysbp_min",
        "h1_sysbp_noninvasive_max",
        "h1_sysbp_noninvasive_min",
        "h1_temp_max",
        "h1_temp_min",
        "h1_albumin_max",
        "h1_albumin_min",
        "h1_bilirubin_max",
        "h1_bilirubin_min",
        "h1_bun_max",
        "h1_bun_min",
        "h1_calcium_max",
        "h1_calcium_min",
        "h1_creatinine_max",
        "h1_creatinine_min",
        "h1_glucose_max",
        "h1_glucose_min",
        "h1_hco3_max",
        "h1_hco3_min",
        "h1_hemaglobin_max",
        "h1_hemaglobin_min",
        "h1_hematocrit_max",
        "h1_hematocrit_min",
        "h1_inr_max",
        "h1_inr_min",
        "h1_lactate_max",
        "h1_lactate_min",
        "h1_platelets_max",
        "h1_platelets_min",
        "h1_potassium_max",
        "h1_potassium_min",
        "h1_sodium_max",
        "h1_sodium_min",
        "h1_wbc_max",
        "h1_wbc_min",
        "h1_arterial_pco2_max",
        "h1_arterial_pco2_min",
        "h1_arterial_ph_max",
        "h1_arterial_ph_min",
        "h1_arterial_po2_max",
        "h1_arterial_po2_min",
        "h1_pao2fio2ratio_max",
        "h1_pao2fio2ratio_min",
    ]
    new_df = df[cols]
    print(f"rows: {new_df.shape[0]}, columns: {new_df.shape[1]}")
    return new_df


# -----Dtype Tools-----
def convert_to_int_col(df):
    cols = ["age", "gcs_eyes_apache", "gcs_motor_apache", "gcs_verbal_apache"]
    for col in cols:
        df[col] = df[col].astype(int)
    return col


def convert_to_bool_col(df):
    cols = [
        "solid_tumor_with_metastasis",
        "lymphoma",
        "leukemia",
        "immunosuppression",
        "hepatic_failure",
        "diabetes_mellitus",
        "aids",
        "cirrhosis",
        "intubated_apache",
        "hospital_death",
        "arf_apache",
    ]
    for col in cols:
        df[col] = df[col].astype(bool)
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
    print(
        f"\t * Number of patients dropped: {rows} \n \t * Number of features dropped:{cols}"
    )
    return df


def fill_with_mode(df):
    cols = ["age", "bmi", "ethnicity", "icu_admit_source"]
    for col in cols:
        if df[col].dtype == float:
            df[col].fillna(int(df[col].mode()), inplace=True)
        else:
            df[col].fillna(df[col].mode()[0], inplace=True)
        print("\t * " + col)
    return df


def fill_with_median(df):
    cols = [
        "d1_diasbp_invasive_max",
        "d1_diasbp_invasive_min",
        "d1_diasbp_max",
        "d1_diasbp_min",
        "d1_diasbp_noninvasive_max",
        "d1_diasbp_noninvasive_min",
        "d1_heartrate_max",
        "d1_heartrate_min",
        "d1_mbp_invasive_max",
        "d1_mbp_invasive_min",
        "d1_mbp_max",
        "d1_mbp_min",
        "d1_mbp_noninvasive_max",
        "d1_mbp_noninvasive_min",
        "d1_resprate_max",
        "d1_resprate_min",
        "d1_spo2_max",
        "d1_spo2_min",
        "d1_sysbp_invasive_max",
        "d1_sysbp_invasive_min",
        "d1_sysbp_max",
        "d1_sysbp_min",
        "d1_sysbp_noninvasive_max",
        "d1_sysbp_noninvasive_min",
        "d1_temp_max",
        "d1_temp_min",
        "d1_albumin_max",
        "d1_albumin_min",
        "d1_bilirubin_max",
        "d1_bilirubin_min",
        "d1_bun_max",
        "d1_bun_min",
        "d1_calcium_max",
        "d1_calcium_min",
        "d1_creatinine_max",
        "d1_creatinine_min",
        "d1_glucose_max",
        "d1_glucose_min",
        "d1_hco3_max",
        "d1_hco3_min",
        "d1_hemaglobin_max",
        "d1_hemaglobin_min",
        "d1_hematocrit_max",
        "d1_hematocrit_min",
        "d1_inr_max",
        "d1_inr_min",
        "d1_lactate_max",
        "d1_lactate_min",
        "d1_platelets_max",
        "d1_platelets_min",
        "d1_potassium_max",
        "d1_potassium_min",
        "d1_sodium_max",
        "d1_sodium_min",
        "d1_wbc_max",
        "d1_wbc_min",
        "d1_arterial_pco2_max",
        "d1_arterial_pco2_min",
        "d1_arterial_ph_max",
        "d1_arterial_ph_min",
        "d1_arterial_po2_max",
        "d1_arterial_po2_min",
        "d1_pao2fio2ratio_max",
        "d1_pao2fio2ratio_min",
        "h1_diasbp_invasive_max",
        "h1_diasbp_invasive_min",
        "h1_diasbp_max",
        "h1_diasbp_min",
        "h1_diasbp_noninvasive_max",
        "h1_diasbp_noninvasive_min",
        "h1_heartrate_max",
        "h1_heartrate_min",
        "h1_mbp_invasive_max",
        "h1_mbp_invasive_min",
        "h1_mbp_max",
        "h1_mbp_min",
        "h1_mbp_noninvasive_max",
        "h1_mbp_noninvasive_min",
        "h1_resprate_max",
        "h1_resprate_min",
        "h1_spo2_max",
        "h1_spo2_min",
        "h1_sysbp_invasive_max",
        "h1_sysbp_invasive_min",
        "h1_sysbp_max",
        "h1_sysbp_min",
        "h1_sysbp_noninvasive_max",
        "h1_sysbp_noninvasive_min",
        "h1_temp_max",
        "h1_temp_min",
        "h1_albumin_max",
        "h1_albumin_min",
        "h1_bilirubin_max",
        "h1_bilirubin_min",
        "h1_bun_max",
        "h1_bun_min",
        "h1_calcium_max",
        "h1_calcium_min",
        "h1_creatinine_max",
        "h1_creatinine_min",
        "h1_glucose_max",
        "h1_glucose_min",
        "h1_hco3_max",
        "h1_hco3_min",
        "h1_hemaglobin_max",
        "h1_hemaglobin_min",
        "h1_hematocrit_max",
        "h1_hematocrit_min",
        "h1_inr_max",
        "h1_inr_min",
        "h1_lactate_max",
        "h1_lactate_min",
        "h1_platelets_max",
        "h1_platelets_min",
        "h1_potassium_max",
        "h1_potassium_min",
        "h1_sodium_max",
        "h1_sodium_min",
        "h1_wbc_max",
        "h1_wbc_min",
        "h1_arterial_pco2_max",
        "h1_arterial_pco2_min",
        "h1_arterial_ph_max",
        "h1_arterial_ph_min",
        "h1_arterial_po2_max",
        "h1_arterial_po2_min",
        "h1_pao2fio2ratio_max",
        "h1_pao2fio2ratio_min",
    ]
    for col in cols:
        df[col].fillna(df[col].median(), inplace=True)
    return df


def drop_rows(df):
    print("\t * Dropping patients with no recorded gender")
    num_patients = df.gender.isna().sum()
    df.dropna(subset=["gender"], inplace=True)
    print(f"\t\t - Number of patients dropped: {num_patients}")
    return df


def fill_with_zero(df):
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
        "hospital_death",
        "ethnicity",
        "gender",
        "age",
        "bmi",
    ]

    for col in cols:
        df[col].fillna(0, inplace=True)
        print("\t * " + col)
    return df


def fill_gcs(df):
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
    min_max = min_max_cols(df)
    for col in min_max:
        vals = df[[col, col.replace("_min", "_max")]].values.copy()

        df[col] = np.nanmin(vals, axis=1)
        df[col.replace("_min", "_max")] = np.nanmax(vals, axis=1)


# Main Function
def get_training_data():
    """
    Reads in the data as a pandas dataframe. Handles null values.
    """
    print("---Acquiring the Data---")
    df = get_raw_data()
    print("Selecting specfic columns to use")
    df = select_columns_to_use(df)
    print("\n")

    print("---Handling Missing Values---")
    print("Filling nulls with mode for the following features:")
    df.pipe(fill_with_mode)
    print("Filling nulls with median")
    df.pipe(fill_with_median)
    print("Handling nulls within rows")
    df.pipe(drop_rows)
    print("Filling nulls with 0 (aka False) for the following columns")
    df.pipe(fill_with_zero)
    print("Filling nulls in the gcs data with the average (5)")
    df.pipe(fill_gcs)
    print("\n")

    print("---Converting Data Types---")
    df.pipe(convert_to_int_col)
    df.pipe(convert_to_bool_col)
    print("done")
    return df
