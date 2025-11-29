# src/models/train_model.py

import logging
import pandas as pd
import numpy as np
import joblib
import json
import os
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor 
from sklearn.metrics import r2_score, mean_squared_error

# Import configuration details
from src.config import (
    TRAIN_DATA_PATH,
    TRAIN_TARGET_PATH,
    FINAL_MODEL_PATH,
    Y_SCALER_PATH,
    FINAL_TRANSFORMER_PATH,
    MODEL_TYPE,
    RANDOM_STATE,
    REPORTS_DIR # Needed for optimized_params.json path
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

# --- Path to Optimized Parameters (Saved by optimize_model.py) ---
OPTIMIZED_PARAMS_PATH = REPORTS_DIR / 'optimized_params.json'


def get_model_estimator(model_type: str, params: dict):
    """
    Initializes and returns the specified model estimator with hyperparameters.
    """
    logger.info(f"Initializing model estimator: {model_type}")
    
    if model_type == 'RandomForestRegressor':
        return RandomForestRegressor(random_state=RANDOM_STATE, **params)
    
    else:
        # Fallback/Error handling
        raise ValueError(f"Unknown model type '{model_type}'. Cannot initialize.")


def load_optimized_params(path: str) -> dict:
    """Loads the best hyperparameters found by the optimization script."""
    try:
        with open(path, 'r') as f:
            params = json.load(f)
        logger.info("Loaded optimized parameters successfully.")
        logger.info(f"Parameters: {params}")
        return params
    except FileNotFoundError:
        logger.warning(f"Optimized parameters file not found at {path}. Using default hyperparameters.")
        # Return sensible default if file is missing (e.g., from config.py)
        return {'n_estimators': 100, 'max_depth': 10, 'random_state': RANDOM_STATE}


def train_pipeline():
    """
    Orchestrates the final model training: loads artifacts, trains the full pipeline, 
    and saves the final model.
    """
    logger.info("--- Starting Final Model Training Pipeline ---")
    
    # 1. Load Data and Artifacts
    try:
        X_train = pd.read_csv(TRAIN_DATA_PATH)
        y_train = pd.read_csv(TRAIN_TARGET_PATH).squeeze()
        preprocessor = joblib.load(FINAL_TRANSFORMER_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
    except FileNotFoundError as e:
        logger.error(f"Required artifact/data not found: {e}. Ensure prior steps are complete.")
        return

    # 2. Load Optimized Parameters
    optimized_params = load_optimized_params(OPTIMIZED_PARAMS_PATH)
    
    # 3. Scale the Target (y) using the fitted Y-Scaler
    y_train_reshaped = y_train.values.reshape(-1, 1)
    y_train_scaled = y_scaler.transform(y_train_reshaped).ravel()
    logger.info("Training target scaled for consistent loss optimization.")

    # 4. Create the Final Model and Full Pipeline
    estimator = get_model_estimator(MODEL_TYPE, optimized_params)
    
    # The full pipeline: Preprocessing (transformer) -> Modeling (estimator)
    full_pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('estimator', estimator)
    ])
    logger.info(f"Full pipeline assembled with {MODEL_TYPE}. Starting final training...")
    #logger.info(f"train Columns :  {X_train.columns}")
    # 5. Train the Pipeline (X_train is unprocessed; pipeline handles scaling)
    full_pipeline.fit(X_train, y_train_scaled)
    logger.info("Training complete.")

    # 6. Save the Final Artifact
    joblib.dump(full_pipeline, FINAL_MODEL_PATH)
    logger.info(f"Final trained model saved successfully to: {FINAL_MODEL_PATH}")
    
    # Optional: Quick check of R2 on training data (not test data!)
    train_score = full_pipeline.score(X_train, y_train_scaled)
    logger.info(f"Quick check (Train R2 Score): {train_score:.4f}")
    
    logger.info("--- Model Training Complete ---")


if __name__ == "__main__":
    # Ensure the y_scaler is created (by optimize_model.py) before running this script
    train_pipeline()