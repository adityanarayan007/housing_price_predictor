# src/models/optimize_model.py

import logging
import pandas as pd
import numpy as np
import json # Used for saving results
import joblib
import os

from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor # Import the final model
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

# Import configuration details
from src.config import (
    TRAIN_DATA_PATH,
    TRAIN_TARGET_PATH,
    Y_SCALER_PATH,
    FINAL_TRANSFORMER_PATH, # The fitted ColumnTransformer artifact
    RANDOM_STATE,
    REPORTS_DIR,
    TEST_SIZE, # Used for logging context
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
if not logger.handlers:
    # 1. Create a Handler to direct output to the console (sys.stderr)
    console_handler = logging.StreamHandler()
    
    # 2. Define the format you want
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_handler.setFormatter(formatter)
    
    # 3. Add the handler to your logger
    logger.addHandler(console_handler)

# --- NEW FUNCTION: Fit and Save the Y-Scaler ---
def fit_and_save_y_scaler(y_path: str, scaler_path: str):
    """
    Fits the StandardScaler on the y_train target and saves the artifact.
    Prevents FileNotFoundError in subsequent steps.
    """
    if os.path.exists(scaler_path):
        logger.info("Y-Scaler artifact already exists. Skipping fit.")
        return
        
    logger.info("Y-Scaler not found. Fitting and saving new Y-Scaler.")
    
    try:
        y_train = pd.read_csv(y_path).squeeze()
    except FileNotFoundError:
        logger.error(f"Training target not found at {y_path}. Cannot fit Y-Scaler.")
        return

    y_scaler = StandardScaler()
    
    # Reshape the target Series to (N, 1) as required by Scikit-learn scalers
    y_train_reshaped = y_train.values.reshape(-1, 1) 
    
    y_scaler.fit(y_train_reshaped) 
    
    joblib.dump(y_scaler, scaler_path)
    logger.info(f"Y-Scaler artifact saved successfully to: {scaler_path}")

# --- 1. Hyperparameter Search Space for SVR ---
def get_search_space() -> dict:
    """
    Defines the randomized search space tailored for the RandomForestRegressor.
    """
    logger.info("Defining RandomForestRegressor hyperparameter search space...")
    
    # Define a wide, strategic search space for the model.
    search_space = {
        # n_estimators: Number of trees to sample
        'n_estimators': [100, 200, 300], 
        
        # max_depth: Limits how deep the trees can grow
        'max_depth': [10, 15, 20, 30, None], # None means unlimited depth
        
        # min_samples_split: The minimum number of samples required to split an internal node
        'min_samples_split': [2, 5, 10], 
        
        # min_samples_leaf: The minimum number of samples required to be at a leaf node
        'min_samples_leaf': [1, 2, 4], 
        
        # max_features: Number of features to consider for best split
        'max_features': [1.0, 'sqrt', 'log2'],
        
        # bootstrap: Whether bootstrap samples are used when building trees
        'bootstrap': [True, False]
    }
    return search_space


def optimize_hyperparams():
    """
    Performs Randomized Search Cross-Validation to find the best SVR hyperparameters 
    and reports the results.
    """
    logger.info("--- Starting SVR Hyperparameter Optimization Pipeline ---")


    fit_and_save_y_scaler(TRAIN_TARGET_PATH, Y_SCALER_PATH)
    # 1. Load Artifacts and Data
    try:
        X_train = pd.read_csv(TRAIN_DATA_PATH)
        y_train = pd.read_csv(TRAIN_TARGET_PATH).squeeze()
        preprocessor = joblib.load(FINAL_TRANSFORMER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
    except FileNotFoundError as e:
        logger.error(f"Required artifacts/data not found: {e}. Ensure prior steps are complete.")
        return

    # 2. Scale the Target (y) using the fitted Y-Scaler
    # SVR is highly sensitive to the magnitude of the target variable.
    y_train_scaled = y_scaler.transform(y_train.values.reshape(-1, 1)).ravel()

    # 3. Create the Full Optimization Pipeline
    # The pipeline combines the fitted feature transformation with the SVR estimator.
    base_model = RandomForestRegressor() 
    
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor), # X-transformation (fitted)
        ('estimator', base_model)       # SVR (unfitted)
    ])
    
    # 4. Define Search and Run Cross-Validation
    param_grid = get_search_space()
    
    # Adjust parameter names for the pipeline: 'C' becomes 'estimator__C', etc.
    param_search_space = {f'estimator__{k}': v for k, v in param_grid.items()}
    
    random_search = RandomizedSearchCV(
        estimator=full_pipeline, 
        param_distributions=param_search_space, 
        n_iter=100,      # <--- High iteration count for best results
        cv=5,            # <--- 5-Fold Cross-Validation for stable performance
        scoring='neg_root_mean_squared_error',
        random_state=42,
        verbose=2,       # Monitor progress
        n_jobs=-1 # Use all available cores
    )
    
    logger.info(f"Starting Randomized Search with 100 iterations and 5-fold CV...")
    # X_train is passed *unprocessed*—the pipeline handles the transformation internally.
    random_search.fit(X_train, y_train_scaled) 

    # 5. Extract and Report Best Parameters
    best_params = random_search.best_params_
    best_score = -random_search.best_score_ # Negate back to positive RMSE (on scaled target)
    
    logger.info(f"\n--- SVR Optimization Complete ---")
    logger.info(f"Best CV RMSE (Scaled Target) found: {best_score:.4f}")
    
    # Clean up the parameter names for saving
    cleaned_params = {k.replace('estimator__', ''): v for k, v in best_params.items()}
    
    # 6. Save Best Parameters (Essential Artifact)
    report_path = REPORTS_DIR / 'optimized_params.json'
    with open(report_path, 'w') as f:
        json.dump(cleaned_params, f, indent=4)

    logger.info(f"Best Parameters saved to: {report_path}")
    logger.info(f"Best Parameters: {cleaned_params}")


if __name__ == "__main__":
    optimize_hyperparams()