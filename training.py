import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, label_binarize
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc, roc_auc_score
from sklearn.multiclass import OneVsRestClassifier
import matplotlib.pyplot as plt
from sklearn import metrics
from itertools import cycle
import warnings
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelEncoder


warnings.filterwarnings('ignore')

# Load the data
data_file_path = './data/lymphography.data'
data = pd.read_csv(data_file_path, header=None)

# Assign column names to the DataFrame
column_names = [
    "class", "lymphatics", "block_of_affere", "bl_of_lymph_c", "bl_of_lymph_s", "by_pass", "extravasates",
    "regeneration_of", "early_uptake_in", "lym_nodes_dimin", "lym_nodes_enlar", "changes_in_lym",
    "defect_in_node", "changes_in_node", "changes_in_stru", "special_forms", "dislocation_of",
    "exclusion_of_no", "no_of_nodes_in"
]
data.columns = column_names

def load_and_preprocess_data(data_file):
    data = pd.read_csv(data_file, header=None)

    # Assuming the class column is the first one (index 0)
    X = data.iloc[:, 1:]
    y = data.iloc[:, 0]

    # Encode the target variable if necessary
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

    return X_train, X_test, y_train, y_test, label_encoder


def train_knn(X_train, y_train, X_test, y_test):
    param_grid = {'n_neighbors': [3, 5, 7, 9, 11]}
    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(knn, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_knn = grid_search.best_estimator_
    y_pred = best_knn.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return best_knn, accuracy, report

def train_random_forest(X_train, y_train, X_test, y_test):
    param_grid = {'n_estimators': [50, 100, 200], 'max_depth': [None, 10, 20, 30]}
    rf = RandomForestClassifier(random_state=42)
    grid_search = GridSearchCV(rf, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return best_rf, accuracy, report

def train_decision_tree(X_train, y_train, X_test, y_test):
    param_grid = {'max_depth': [None, 10, 20, 30]}
    dt = DecisionTreeClassifier(random_state=42)
    grid_search = GridSearchCV(dt, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_dt = grid_search.best_estimator_
    y_pred = best_dt.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return best_dt, accuracy, report

def train_gradient_boosting(X_train, y_train, X_test, y_test):
    param_grid = {'n_estimators': [50, 100, 200], 'learning_rate': [0.01, 0.1, 0.2]}
    gb = GradientBoostingClassifier(random_state=42)
    grid_search = GridSearchCV(gb, param_grid, cv=5)
    grid_search.fit(X_train, y_train)
    best_gb = grid_search.best_estimator_
    y_pred = best_gb.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred)
    return best_gb, accuracy, report

if __name__ == "__main__":
    # File paths
    data_file = './data/lymphography.data'

    # Load and preprocess data
    X_train, X_test, y_train, y_test, label_encoder = load_and_preprocess_data(data_file)

    # Train and evaluate KNN
    knn_model, knn_accuracy, knn_report = train_knn(X_train, y_train, X_test, y_test)
    print(f'KNN Accuracy: {knn_accuracy:.2f}')
    print(f'KNN Classification Report:\n{knn_report}')

    # Train and evaluate Random Forest
    rf_model, rf_accuracy, rf_report = train_random_forest(X_train, y_train, X_test, y_test)
    print(f'Random Forest Accuracy: {rf_accuracy:.2f}')
    print(f'Random Forest Classification Report:\n{rf_report}')

    # Train and evaluate Decision Tree
    dt_model, dt_accuracy, dt_report = train_decision_tree(X_train, y_train, X_test, y_test)
    print(f'Decision Tree Accuracy: {dt_accuracy:.2f}')
    print(f'Decision Tree Classification Report:\n{dt_report}')

    # Train and evaluate Gradient Boosting
    gb_model, gb_accuracy, gb_report = train_gradient_boosting(X_train, y_train, X_test, y_test)
    print(f'Gradient Boosting Accuracy: {gb_accuracy:.2f}')
    print(f'Gradient Boosting Classification Report:\n{gb_report}')
