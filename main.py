# Imports 
import pandas as pd 
import numpy as np 

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

import matplotlib.pyplot as plt 
import seaborn as sns

# Load Titanic dataset
data = pd.read_csv("titanic.csv")
data.info()
print(data.isnull().sum())

# Data Cleaning and Feature Engineering 
def preprocess_data(df):
    # Drop unnecessary columns
    df = df.drop(columns=["PassengerId", "Name", "Ticket", "Cabin"])
    
    # Fill missing Embarked values and drop the column
    df["Embarked"] = df["Embarked"].fillna("S")
    df = df.drop(columns=["Embarked"])
    
    # Fill missing ages
    fill_missing_ages(df)
    
    # Convert Gender to numeric
    df["Sex"] = df["Sex"].map({'male': 1, "female": 0})
    
    # Create new features: FamilySize, IsAlone, FareBin, AgeBin
    df["FamilySize"] = df["SibSp"] + df["Parch"]
    df["IsAlone"] = np.where(df["FamilySize"] == 0, 1, 0)
    df["FareBin"] = pd.qcut(df["Fare"], 4, labels=False)
    df["AgeBin"] = pd.cut(df["Age"], bins=[0, 12, 20, 40, 60, np.inf], labels=False)
    
    return df

# Fill missing ages based on median age of each Pclass
def fill_missing_ages(df):
    age_fill_map = {}
    for pclass in df["Pclass"].unique():
        if pclass not in age_fill_map:
            age_fill_map[pclass] = df[df["Pclass"] == pclass]["Age"].median()
            
    df["Age"] = df.apply(lambda row: age_fill_map[row["Pclass"]] if pd.isnull(row["Age"]) else row["Age"], axis=1)

# Preprocess the data
data = preprocess_data(data)

# Create Features (X) and Target (y)
X = data.drop(columns=["Survived"])
y = data["Survived"]

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)

# Scale features using MinMaxScaler
scaler = MinMaxScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Perform hyperparameter tuning for KNN using GridSearchCV
def tune_model(X_train, y_train):
    param_grid = {
        "n_neighbors":range(1,21),  # Test different numbers of neighbors
        "metric" : ["euclidean", "manhattan", "minkowski"],  # Test distance metrics
        "weights" : ["uniform","distance"]  # Test weight strategies
    }
    
    model = KNeighborsClassifier()
    grid_search = GridSearchCV(model, param_grid, cv=5, n_jobs=-1)  # Perform grid search
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_  # Return the best model

# Get the best KNN model
best_model = tune_model(X_train, y_train)

# Evaluate the model's accuracy and confusion matrix
def evaluate_model(model, X_test, y_test):
    prediction = model.predict(X_test)  # Make predictions
    accuracy = accuracy_score(y_test, prediction)  # Calculate accuracy
    matrix = confusion_matrix(y_test, prediction)  # Generate confusion matrix
    return accuracy, matrix

# Evaluate the best model
accuracy, matrix = evaluate_model(best_model, X_test, y_test)

# Print accuracy and confusion matrix
print(f'Accuracy: {accuracy*100:.2f}%')
print(f'Confusion Matrix:')
print(matrix)

# Plot confusion matrix using Seaborn
def plot_model(matrix):
    plt.figure(figsize=(10,7))
    sns.heatmap(matrix, annot=True, fmt="d", xticklabels=["Survived","Not Survived"], yticklabels=["Not Survived", "Survived"])
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted Value")
    plt.ylabel("True Values")
    plt.show()

# Plot the confusion matrix
plot_model(matrix)