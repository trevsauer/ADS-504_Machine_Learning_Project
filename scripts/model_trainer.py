import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt 
from sklearn import RandomOverSampler, RandomUnderSampler
from sklearn import StandardScaler, MinMaxScaler
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.metrics import accuracy_score, precision_score, roc_auc_score
from sklearn.metrics import recall_score, confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE, ADASYN


# Define a dictionary mapping sampling techniques to their corresponding resampling objects
resampling_methods = {
    'oversample': RandomOverSampler,
    'smote': SMOTE,
    'adasyn': ADASYN,
    'undersample': RandomUnderSampler
}

def train_and_evaluate_classifier(classifier, X_train, y_train, X_test, y_test, 
                                  scaling_method=None, sampling_technique=None, threshold=0.5):
    start_time = time.time()

    if sampling_technique in resampling_methods:
        resampler = resampling_methods[sampling_technique](sampling_strategy='auto', random_state=42)
        X_train, y_train = resampler.fit_resample(X_train, y_train)

    if scaling_method == 'standard':
        scaler = StandardScaler()
    elif scaling_method == 'minmax':
        scaler = MinMaxScaler()
    else:
        scaler = None

    if scaler is not None:
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    classifier.fit(X_train, y_train)
    end_time = time.time()
    elapsed_time = end_time - start_time

    # Obtain predicted probabilities for the positive class
    y_pred_probs = classifier.predict_proba(X_test)[:, 1]

    # Adjust the threshold to convert probabilities to binary predictions
    y_pred = (y_pred_probs > threshold).astype(int)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = conf_matrix.ravel()
    specificity = tn / (tn + fp)
    roc_auc = roc_auc_score(y_test, y_pred_probs)

    feature_importances = None
    if hasattr(classifier, 'feature_importances_'):
        feature_importances = pd.Series(classifier.feature_importances_, index=X.columns)
        
    return accuracy, precision, recall, specificity, roc_auc, elapsed_time, feature_importances, y_pred_probs


X = final_df.drop(columns=['Potability'])
y = final_df['Potability']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.3, random_state=42)

# Initialize classifiers
classifiers = {
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    # 'SVM': SVC(kernel='linear', random_state=42), # Scale data before using SVM
    'XGBoost': XGBClassifier(random_state=42),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100,), random_state=42),
    'Logistic Regression': LogisticRegression(random_state=42),
    'K-Nearest Neighbors': KNeighborsClassifier(),
    'AdaBoost': AdaBoostClassifier(random_state=42),
    'Naive Bayes': GaussianNB(),
    'Bagging Classifier': BaggingClassifier(base_estimator=DecisionTreeClassifier(), random_state=42),
    # 'Ridge Classifier': RidgeClassifier(random_state=42),
    'Quadratic Discriminant Analysis': QuadraticDiscriminantAnalysis(),
    'Linear Discriminant Analysis': LinearDiscriminantAnalysis()
}

# Train and evaluate each classifier
results = {}

# To enable scaling, pass scaling_method='standard' or scaling_method='minmax'
# To disable scaling, pass scaling_method=None
# To use sampling, pass sampling_technique='oversample', 'smote', 'adasyn', or 'undersample'
# To use both scaling and sampling, pass both scaling_method and sampling_technique

# Train and evaluate each classifier with the adjusted threshold
threshold = 0.5  # Set the desired threshold value (can be any value between 0 and 1)

for name, classifier in classifiers.items():
    accuracy, precision, recall, specificity, roc_auc, elapsed_time, feature_importances, _ = train_and_evaluate_classifier(
        classifier, X_train, y_train, X_test, y_test, scaling_method=None, sampling_technique=None, threshold=threshold)
    
    results[name] = {
        'Accuracy': accuracy,
        'Precision': precision,
        'Sensitivity/Recall': recall,
        'Specificity': specificity,
        'ROC AUC': roc_auc,
        'Elapsed Time': elapsed_time,
        'Feature Importances': feature_importances
    }

    # Plot feature importances
    if feature_importances is not None:
        plt.figure(figsize=(10, 6))
        feature_importances.plot(kind='barh')
        plt.xlabel('Feature Importance')
        plt.title(f'{name} - Feature Importances')
        plt.show()

# Print results without numerical values
for name, metrics in results.items():
    print(f"{name} - Evaluation Metrics:")
    for metric, value in metrics.items():
        if metric == 'Feature Importances':
            continue  # Skip printing feature importances
        if metric == 'Elapsed Time':
            print(f"{metric}: {value:.4f} seconds")
        else:
            print(f"{metric}: {value}")
    print()