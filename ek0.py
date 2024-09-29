import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.metrics import classification_report, confusion_matrix
import tkinter as tk
from tkinter import filedialog
import numpy

# Function to open a file dialog and select a CSV file
def load_file():
    root = tk.Tk()
    root.withdraw()  # Close the root window
    file_path = filedialog.askopenfilename(title="Select file", filetypes=[("CSV files", "*.csv")])
    return file_path

# 1. Load data, print column names and dataset size
file_path = load_file()
if file_path:  # Ensure a file was selected
    df = pd.read_csv(file_path)
    print("Column names:", df.columns)
    print("Dataset size:", df.shape)
else:
    print("No file selected. Exiting the script.")
    exit()

# 2. Handle missing values (fill or drop if possible)
print(df.isnull().sum())  # Check the number of missing values in each column

# Fill missing values only in numerical columns
df.fillna(df.select_dtypes(include=['float64', 'int64']).mean(), inplace=True)

# 3. Data visualization: correlation heatmap and histograms
# Select only numerical columns for correlation matrix
numerical_df = df.select_dtypes(include=['float64', 'int64'])

plt.figure(figsize=(12, 8))
correlation_matrix = numerical_df.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title("Feature Correlation Matrix")
plt.show()

# Histograms of feature distributions
df.hist(bins=30, figsize=(12, 10))
plt.suptitle("Feature Distribution Histograms")
plt.show()

# Automatically detect target and features based on the dataset

# Detect target variable (categorical column with few unique values)
def detect_target(df, max_unique_values=10):
    for column in df.columns:
        if df[column].dtype == 'object' or df[column].nunique() <= max_unique_values:
            return column
    return None

# Detect features (numerical columns)
def detect_features(df, exclude_column):
    features = []
    for column in df.columns:
        if column != exclude_column and df[column].dtype in ['float64', 'int64']:
            features.append(column)
    return features

# Detect target and features from the dataset
target = detect_target(df)
features = detect_features(df, target)

if target is not None and len(features) > 0:
    print(f"Detected target: {target}")
    print(f"Detected features: {features}")
else:
    print("Failed to detect target or features.")

# Proceed with the rest of the code using detected target and features
# Boxplots of features relative to the target variable
for feature in features:
    if feature in df.columns:
        sns.boxplot(x=target, y=feature, data=df)
        plt.title(f"Boxplot of {feature} vs {target}")
        plt.show()
    else:
        print(f"Feature {feature} not found in dataframe columns")

# Boxplots of features relative to the target variable
for feature in features:
    if feature in df.columns:
        sns.boxplot(x=target, y=feature, data=df)
        plt.title(f"Boxplot of {feature} vs {target}")
        plt.show()
    else:
        print(f"Feature {feature} not found in dataframe columns")

# 4. Normalize data
# Ensure only numerical columns are included for normalization
features_df = df[features]

# Drop columns with non-numeric data before normalization
numerical_features_df = features_df.select_dtypes(include=['float64', 'int64'])

scaler = StandardScaler()
scaled_features = scaler.fit_transform(numerical_features_df)  # Normalize features
df_scaled = pd.DataFrame(scaled_features, columns=numerical_features_df.columns)  # Create a new DataFrame

# 5. Train classifiers
X_train, X_test, y_train, y_test = train_test_split(df_scaled, df[target], test_size=0.3, random_state=42)

# kNN
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)
print("KNN:")
print(classification_report(y_test, y_pred_knn))
print(confusion_matrix(y_test, y_pred_knn))

# Decision Tree
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)
print("Decision Tree:")
print(classification_report(y_test, y_pred_dt))
print(confusion_matrix(y_test, y_pred_dt))

# SVM with GridSearchCV for parameter tuning
param_grid_svm = {'C': [0.1, 1, 10], 'gamma': [1, 0.1, 0.01]}
grid_svm = GridSearchCV(SVC(), param_grid_svm, refit=True, verbose=2)
grid_svm.fit(X_train, y_train)
y_pred_svm = grid_svm.predict(X_test)
print("SVM:")
print(classification_report(y_test, y_pred_svm))
print(confusion_matrix(y_test, y_pred_svm))

# Random Forest
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
print("Random Forest:")
print(classification_report(y_test, y_pred_rf))
print(confusion_matrix(y_test, y_pred_rf))

# AdaBoost
ada = AdaBoostClassifier(algorithm='SAMME')
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)
print("AdaBoost:")
print(classification_report(y_test, y_pred_ada, labels=numpy.unique(y_pred_ada)))
print(confusion_matrix(y_test, y_pred_ada))

# Parameter tuning for kNN
param_grid_knn = {'n_neighbors': list(range(1, 31))}
grid_knn = GridSearchCV(KNeighborsClassifier(), param_grid_knn, refit=True)
grid_knn.fit(X_train, y_train)
print("Optimal n_neighbors for kNN:", grid_knn.best_params_)