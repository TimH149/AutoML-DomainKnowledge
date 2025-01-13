import os 
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import KNNImputer
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

def preprocess_data(X):
    for column in X.columns:
        if X[column].dtype == 'object':
            # Apply LabelEncoder 
            X[column] = LabelEncoder().fit_transform(X[column].astype(str))
        elif X[column].dtype == 'category':
            # Tranform to numeric values
            X[column] = X[column].cat.codes
    # Apply KNN imputation for missing values
    imputer = KNNImputer(n_neighbors=5) 
    X = imputer.fit_transform(X)
    return X

def is_linearly_separable(X, y, tolerance=0.5, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42)
    
    classes = np.unique(y)
    separable = True

    # One-vs-All 
    for cls in classes:
        # Create OvA labels for this class
        y_train_binary = (y_train == cls).astype(int)
        y_test_binary = (y_test == cls).astype(int)
        
        # Initialize and train a soft margin SVM 
        model = LinearSVC(max_iter = 10000, C = 1.0)
        model.fit(X_train, y_train)

        if model.n_iter_ == model.max_iter:
            # If the model didn't converge, return False
            return False

        # Make predictions and calculate accuracy 
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)

        # Check if predictions are below threshold 
        if accuracy < tolerance:
            separable = False
            return separable
            break

    return separable

exclude = pd.read_csv("ExcludeDatasets.csv", delimiter = ";")
ExcludeDatasets = list(exclude["Dataset"])

# Define a path to the dataset directory
save_path = os.path.expanduser('~/openml_benchmark_data_classification99/')
dataset_dirs = [d for d in os.listdir(save_path) if os.path.isdir(os.path.join(save_path, d))]


LinearSep = {
    'Dataset': [],
    'Linear_sep': []
}

for dataset_dir in dataset_dirs:

    # Skip datasets that should be excluded (e.g. based on their size or complexity)
    if dataset_dir in ExcludeDatasets:
        continue

    print(dataset_dir)

    features_path = os.path.join(save_path, dataset_dir, 'features.csv')
    target_path = os.path.join(save_path, dataset_dir, 'target.csv')

    X = preprocess_data(pd.read_csv(features_path))
    y = pd.read_csv(target_path).values.ravel()
    
    lin_sep = is_linearly_separable(X,y)

    LinearSep['Dataset'].append(dataset_dir)
    LinearSep['Linear_sep'].append(lin_sep)


pd.DataFrame(LinearSep).to_csv("LinearDatasets.csv", index = False)
    

    