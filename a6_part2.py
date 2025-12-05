"""
Assignment 6 Part 2: House Price Prediction (Multivariable Regression)

This assignment predicts house prices using MULTIPLE features.
Complete all the functions below following the in-class car price example.
"""

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np


def load_and_explore_data(filename):
    """
    Load the house price data and explore it
    
    Args:
        filename: name of the CSV file to load
    
    Returns:
        pandas DataFrame containing the data
    """

    data = pd.read_csv(filename)

    print("=== House Price Data ===")
    print("\nFirst 5 rows:")
    print(data.head())

    print(f"\nDataset shape: {data.shape[0]} rows, {data.shape[1]} columns")

    print("\nBasic statistics:")
    print(data.describe())

    print("\nColumn names:", list(data.columns))

    return data


def visualize_features(data):
    """
    Create 4 scatter plots (one for each feature vs Price)
    """

    plt.figure(figsize=(12, 10))
    plt.suptitle("House Features vs Price")

    # --- Plot 1: SquareFeet ---
    plt.subplot(2, 2, 1)
    plt.scatter(data["SquareFeet"], data["Price"], color="blue", alpha=0.6)
    plt.xlabel("Square Feet")
    plt.ylabel("Price")
    plt.title("SquareFeet vs Price")
    plt.grid(True)

    # --- Plot 2: Bedrooms ---
    plt.subplot(2, 2, 2)
    plt.scatter(data["Bedrooms"], data["Price"], color="green", alpha=0.6)
    plt.xlabel("Bedrooms")
    plt.ylabel("Price")
    plt.title("Bedrooms vs Price")
    plt.grid(True)

    # --- Plot 3: Bathrooms ---
    plt.subplot(2, 2, 3)
    plt.scatter(data["Bathrooms"], data["Price"], color="red", alpha=0.6)
    plt.xlabel("Bathrooms")
    plt.ylabel("Price")
    plt.title("Bathrooms vs Price")
    plt.grid(True)

    # --- Plot 4: Age ---
    plt.subplot(2, 2, 4)
    plt.scatter(data["Age"], data["Price"], color="orange", alpha=0.6)
    plt.xlabel("Age")
    plt.ylabel("Price")
    plt.title("Age vs Price")
    plt.grid(True)

    plt.tight_layout()
    plt.savefig("feature_plots.png", dpi=300)
    plt.show()


def prepare_features(data):
    """
    Separate features (X) from target (y)
    """

    feature_columns = ["SquareFeet", "Bedrooms", "Bathrooms", "Age"]

    X = data[feature_columns]
    y = data["Price"]

    print("\n=== Feature Preparation ===")
    print(f"Features (X) shape: {X.shape}")
    print(f"Target (y) shape: {y.shape}")
    print("Feature columns:", feature_columns)

    return X, y


def split_data(X, y):
    """
    Split data into training and testing sets
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("\n=== Data Split ===")
    print(f"Training set: {len(X_train)} samples")
    print(f"Testing set: {len(X_test)} samples")

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train, feature_names):
    """
    Train a multivariable linear regression model
    """

    model = LinearRegression()
    model.fit(X_train, y_train)

    print("\n=== Model Training ===")
    print(f"Intercept: {model.intercept_:.2f}")

    print("\nCoefficients:")
    for name, coef in zip(feature_names, model.coef_):
        print(f"  {name}: {coef:.2f}")

    # Print full equation
    equation = "Price = "
    for name, coef in zip(feature_names, model.coef_):
        equation += f"({coef:.2f} × {name}) + "
    equation += f"{model.intercept_:.2f}"

    print("\nFull Model Equation:")
    print(equation)

    return model


def evaluate_model(model, X_test, y_test, feature_names):
    """
    Evaluate the model's performance
    """

    predictions = model.predict(X_test)

    r2 = r2_score(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = np.sqrt(mse)

    print("\n=== Model Evaluation ===")
    print(f"R² Score: {r2:.4f}  (closer to 1 = better)")
    print(f"RMSE: {rmse:.2f} dollars (average prediction error)")

    # Feature Importance
    importances = np.abs(model.coef_)
    sorted_idx = np.argsort(importances)[::-1]

    print("\nFeature Importance (highest → lowest):")
    for idx in sorted_idx:
        print(f"  {feature_names[idx]}: {model.coef_[idx]:.2f}")

    return predictions


def compare_predictions(y_test, predictions, num_examples=5):
    """
    Show side-by-side comparison of actual vs predicted prices
    """

    print("\n=== Prediction Comparison ===")
    print(f"{'Actual':>10} | {'Predicted':>12} | {'Error':>10} | {'% Error':>10}")
    print("-" * 55)

    for actual, pred in list(zip(y_test, predictions))[:num_examples]:
        error = actual - pred
        pct_error = abs(error) / actual * 100
        print(
            f"${actual:>8,.0f} | ${pred:>10,.0f} | ${error:>8,.0f} | {pct_error:>8.2f}%"
        )


def make_prediction(model, sqft, bedrooms, bathrooms, age):
    """
    Make a prediction for a specific house
    """

    new_data = pd.DataFrame(
        [[sqft, bedrooms, bathrooms, age]],
        columns=["SquareFeet", "Bedrooms", "Bathrooms", "Age"]
    )

    predicted_price = model.predict(new_data)[0]

    print("\n=== New House Prediction ===")
    print(f"Square Feet: {sqft}")
    print(f"Bedrooms: {bedrooms}")
    print(f"Bathrooms: {bathrooms}")
    print(f"Age: {age} years")
    print(f"\nPredicted Price: ${predicted_price:,.2f}")

    return predicted_price


if __name__ == "__main__":
    print("=" * 70)
    print("HOUSE PRICE PREDICTION - YOUR ASSIGNMENT")
    print("=" * 70)

    # Step 1: Load and explore
    data = load_and_explore_data("house_prices.csv")

    # Step 2: Visualize features
    visualize_features(data)

    # Step 3: Prepare features
    X, y = prepare_features(data)

    # Step 4: Split data
    X_train, X_test, y_train, y_test = split_data(X, y)

    # Step 5: Train model
    model = train_model(X_train, y_train, X.columns)

    # Step 6: Evaluate model
    predictions = evaluate_model(model, X_test, y_test, X.columns)

    # Step 7: Compare predictions
    compare_predictions(y_test, predictions, num_examples=10)

    # Step 8: Make a new prediction (example house)
    make_prediction(model, sqft=1800, bedrooms=3, bathrooms=2, age=10)

    print("\n" + "=" * 70)
    print("✓ Assignment complete! Check your saved plots.")
    print("Don't forget to complete a6_part2_writeup.md!")
