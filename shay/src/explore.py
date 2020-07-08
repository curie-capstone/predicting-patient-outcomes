from scipy import stats
import pandas as pd

def significance_testing_of_nulls(df):
    # Set alpha value
    alpha = 0.05

    for col in df.columns:

        a, b = df[col], df["hospital_death"]

        observed = pd.crosstab(a, b) 
        chi2, p, degf, expected = stats.chi2_contingency(observed)

        if p < alpha:
            # Reject the null hypothesis
            print("({} and hospital_death) are  dependent of each other. (p = {})".format(col, p))
        else:
            # Failed to reject the null hypothesis
            print("({} and hospital_death) are  independent of each other. (p = {})".format(col, p))