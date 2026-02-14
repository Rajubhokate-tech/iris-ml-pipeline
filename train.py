import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import joblib

# Part 1: Load and Explore
iris = load_iris()
X = iris.data
y = iris.target

# Part 2: Train Model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Save the model
joblib.dump(model, 'iris_model.pkl')
print("Model trained and saved successfully!")
