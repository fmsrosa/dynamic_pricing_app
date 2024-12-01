import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import joblib

SCALER: StandardScaler = StandardScaler()

def load_data(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Load the dataset and return features (X) and target (y).
    """
    data = pd.read_csv(file_path).head(100)
    X = data[["stars", "reviews", "boughtInLastMonth"]].values
    y = data["price"].values
    return X, y

def preprocess_data(X: np.ndarray, X_train: bool = True) -> np.ndarray:
    """
    Preprocess the data (scaling).
    Apply fit_transform on training data and transform on test data.
    """
    
    if X_train:
        # Fit and transform the training data
        return SCALER.fit_transform(X)
    else:
        # Only transform the test data
        return SCALER.transform(X)

def train_model(X_train: np.ndarray, y_train: np.ndarray) -> LinearRegression:
    """
    Train a Linear Regression model on the scaled training data.
    """
    model = LinearRegression()
    model.fit(X_train, y_train)
    return model

def evaluate_model(model: LinearRegression, X_test: np.ndarray, y_test: np.ndarray) -> float:
    """
    Evaluate the trained model using the test data and return the R² score.
    """
    return model.score(X_test, y_test)

def save_model(pipeline: Pipeline, filename: str) -> None:
    """
    Save the trained pipeline to a file.
    """
    joblib.dump(pipeline, filename)

def main():
    # Load the data
    X, y = load_data("data/raw/amz_us_price_prediction_dataset.csv")
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
    
    # Preprocess the data (scaling)
    X_train_scaled = preprocess_data(X_train, X_train=True)
    X_test_scaled = preprocess_data(X_test, X_train=False)

    # Train the model
    model = train_model(X_train_scaled, y_train)

    # Evaluate the model
    score = evaluate_model(model, X_test_scaled, y_test)
    print(f"Model score: {score}")

    # Combine scaler and model into a pipeline
    pipeline = Pipeline([
        ('scaler', SCALER),
        ('model', model)
    ])

    # Save the trained pipeline
    save_model(pipeline, 'model/linear_regression_pricing_model.pkl')

if __name__ == "__main__":
    main()

