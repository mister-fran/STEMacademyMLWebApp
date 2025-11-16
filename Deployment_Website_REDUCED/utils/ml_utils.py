"""
Machine Learning utilities for the ML Academy app
"""
import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from lightgbm import LGBMRegressor
from .config import MODEL_CONFIG

def prepare_data(dataset, target_column, selected_columns=None):
    """
    Prepare data for machine learning
    
    Args:
        dataset: pandas DataFrame
        target_column: name of target column
        selected_columns: list of columns to use as features
    
    Returns:
        X_train, X_test, y_train, y_test
    """
    if selected_columns:
        # Filter dataset to only include selected columns
        dataset_filtered = dataset[selected_columns]
        X = dataset_filtered.drop(columns=[target_column])
    else:
        X = dataset.drop(columns=[target_column])
    
    y = dataset[target_column]
    
    # Handle missing values
    X = X.fillna(0)
    y = y.fillna(0)
    
    # Split data
    return train_test_split(
        X, y, 
        test_size=MODEL_CONFIG["test_size"], 
        random_state=MODEL_CONFIG["random_state"]
    )

def train_lgbm_model(X_train, y_train):
    """Train a LightGBM regression model"""
    with st.spinner("Træner model..."):
        model = LGBMRegressor(random_state=MODEL_CONFIG["random_state"])
        model.fit(X_train, y_train)
    st.success("Model trænet!")
    return model

def evaluate_model(model, X_test, y_test, error_metric="MSE"):
    """Evaluate the trained model"""
    y_pred = model.predict(X_test)
    
    if error_metric == "MSE":
        error = mean_squared_error(y_test, y_pred)
        error_name = "Mean Squared Error"
    elif error_metric == "MAE":
        error = mean_absolute_error(y_test, y_pred)
        error_name = "Mean Absolute Error"
    else:
        error = mean_squared_error(y_test, y_pred)
        error_name = "Mean Squared Error"
    
    return y_pred, error, error_name

def create_predictions_dataframe(X_test, y_test, y_pred, target_column):
    """Create a dataframe with predictions"""
    predictions_df = X_test.copy()
    predictions_df[target_column] = y_test.values
    predictions_df['Forudsigelse'] = y_pred
    return predictions_df

def display_code_example(target_column, error_metric="MSE"):
    """Display code example for the regression"""
    code = f"""
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Prepare data
X = dataset.drop(columns=['{target_column}'])
y = dataset['{target_column}']

# Handle missing values
X = X.fillna(0)
y = y.fillna(0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = LGBMRegressor(random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
"""
    
    if error_metric == "MSE":
        code += """
error = mean_squared_error(y_test, y_pred)
print(f"Mean Squared Error: {error}")
"""
    elif error_metric == "MAE":
        code += """
error = mean_absolute_error(y_test, y_pred)
print(f"Mean Absolute Error: {error}")
"""
    
    return code