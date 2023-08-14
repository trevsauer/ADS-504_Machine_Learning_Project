import pandas as pd
from sklearn.impute import KNNImputer
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import LocalOutlierFactor

def preprocess_data(data, missing_method=None, outlier_method=None, knn_neighbors=5, lof_neighbors=20, lof_contamination=0.1):
    preprocessed_data = data.copy()
    
    if missing_method == 'drop':
        preprocessed_data.dropna(inplace=True)
    elif missing_method == 'knn':
        imputer = KNNImputer(n_neighbors=knn_neighbors)
        preprocessed_data = pd.DataFrame(imputer.fit_transform(preprocessed_data), columns=data.columns)
    elif missing_method == 'regression':
        for column in preprocessed_data.columns:
            incomplete_rows = preprocessed_data[preprocessed_data[column].isnull()]
            complete_rows = preprocessed_data[~preprocessed_data[column].isnull()]
            X_complete = complete_rows.drop(columns=[column])
            y_complete = complete_rows[column]
            regressor = RandomForestRegressor()  # Use any suitable regression model
            regressor.fit(X_complete, y_complete)
            imputed_values = regressor.predict(incomplete_rows.drop(columns=[column]))
            preprocessed_data.loc[preprocessed_data[column].isnull(), column] = imputed_values
    
    if outlier_method == 'lof' and missing_method == 'knn':
        lof = LocalOutlierFactor(n_neighbors=lof_neighbors, contamination=lof_contamination)
        outlier_scores = lof.fit_predict(preprocessed_data)
        preprocessed_data = preprocessed_data[outlier_scores == 1]
    
    return preprocessed_data

# Example usage
# Replace 'water' with your actual DataFrame
preprocessed_data = preprocess_data(water, missing_method='knn', outlier_method='lof', knn_neighbors=5, lof_neighbors=20, lof_contamination=0.1)

# Display the preprocessed data
print(preprocessed_data)
