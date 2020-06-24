import pandas as pd

def get_feature_importance(X, feature_importance):
    '''
    Creates a Dataframe displaying the feature and what importance the model gave for that feature (the weight)
    '''
    
    feature_list = []
    importance_list = []
    for feature, importance in zip(X, feature_importance):
        if importance > 0:
            feature_list.append(feature)
            importance_list.append(importance)
    features_series = pd.Series(feature_list)
    importance_series = pd.Series(importance_list)
    df = pd.DataFrame(data=[features_series, importance_series])
    df = df.T
    df.columns = ['feature', 'importance']
    return df.drop_duplicates().sort_values(by='importance', ascending=False)
