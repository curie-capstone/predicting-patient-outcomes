import pandas as pd
from scipy import stats

def get_numeric(df):
    numeric_col_list = []
    for col in df.columns:
        if df[col].dtype in [int, float]:
            numeric_col_list.append(col)
    return numeric_col_list


def aggregate_test(col1, col2, target_col):
    