import pandas as pd
import numpy as np

# Data Preprocessing Functions
def mean_impute(column):
    mean_value = np.nanmean(column)
    return np.where(np.isnan(column), mean_value, column)

def label_encode(column):
    unique_values = list(set(column))
    mapping = {val: idx for idx, val in enumerate(unique_values)}
    return np.array([mapping[val] for val in column]), mapping

def normalize(column):
    column = column.astype(np.float64)
    if np.max(column) - np.min(column) == 0:
        return np.zeros_like(column)
    return (column - np.min(column)) / (np.max(column) - np.min(column))

def preprocess_data(df):
    df = df.drop(columns=['PassengerId', 'Name', 'Ticket', 'Cabin'], errors='ignore')
    df['Age'] = mean_impute(df['Age'].values)
    df['Fare'] = mean_impute(df['Fare'].values)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode()[0])
    df['Sex'], _ = label_encode(df['Sex'].values)
    embarked_dummies = pd.get_dummies(df['Embarked'], prefix='Embarked', drop_first=True)
    df = pd.concat([df.drop(columns=['Embarked']), embarked_dummies], axis=1)
    return df

# Logistic Regression Model (Implemented from Scratch)
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def initialize_weights(n_features):
    return np.zeros(n_features, dtype=np.float64), 0.0

def train_logistic_regression(X, y, lr=0.01, epochs=1000):
    m, n = X.shape
    weights, bias = initialize_weights(n)
    for _ in range(epochs):
        linear_model = np.dot(X, weights) + bias
        predictions = sigmoid(linear_model)
        gradient_w = np.dot(X.T, (predictions - y)) / m
        gradient_b = np.sum(predictions - y) / m
        weights = weights - lr * gradient_w
        bias = bias - lr * gradient_b
    return weights, bias

def predict_logistic_regression(X, weights, bias):
    linear_model = np.dot(X, weights) + bias
    predictions = sigmoid(linear_model)
    return (predictions >= 0.5).astype(int)

def evaluate_model(y_true, y_pred):
    accuracy = np.sum(y_true == y_pred) / len(y_true)
    print("Accuracy:", accuracy)

# Load and Process Data
test_path = "Dataset/tested.csv"
df = pd.read_csv(test_path)
df['Survived'] = df['Survived'].fillna(0).astype(int)
df = preprocess_data(df)

X = df.drop(columns=['Survived']).values.astype(np.float64)
y = df['Survived'].values.astype(np.float64)
X = np.apply_along_axis(normalize, 0, X)

# Train and Predict with Logistic Regression
weights, bias = train_logistic_regression(X, y, lr=0.01, epochs=5000)
y_pred = predict_logistic_regression(X, weights, bias)

evaluate_model(y, y_pred)
print("Predictions:", y_pred)