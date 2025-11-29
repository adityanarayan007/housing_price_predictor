# src/models/evaluate_model.py

import logging
import pandas as pd
import numpy as np
import json
import joblib
from pathlib import Path
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Any

# Import configuration and artifact paths
from src.config import (
    TEST_DATA_PATH,
    TEST_TARGET_PATH,
    FINAL_MODEL_PATH,
    Y_SCALER_PATH,
    REPORTS_DIR 
)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --- Ensure logging is set up for standalone execution ---
if not logger.handlers:
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logger = logging.getLogger(__name__)


def load_artifacts_and_data() -> tuple[Any, pd.DataFrame, np.ndarray, Any]:
    """Loads the model pipeline, test data, and y-scaler from disk."""
    logger.info("Loading final model, test data, and y-scaler.")
    try:
        # Load artifacts
        model = joblib.load(FINAL_MODEL_PATH)
        y_scaler = joblib.load(Y_SCALER_PATH)
        
        # Load test data
        X_test = pd.read_csv(TEST_DATA_PATH)
        y_test = pd.read_csv(TEST_TARGET_PATH).squeeze() # Target is a Series of original prices

        return model, X_test, y_test.values, y_scaler
    except FileNotFoundError as e:
        logger.error(f"Required file not found: {e}. Ensure training and optimization were run.")
        raise


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Calculates key regression metrics (R2, RMSE, MAE)."""
    
    metrics = {
        'R2_Score': r2_score(y_true, y_pred),
        'Mean_Squared_Error': mean_squared_error(y_true, y_pred),
        'Root_Mean_Squared_Error': np.sqrt(mean_squared_error(y_true, y_pred)),
        'Mean_Absolute_Error': mean_absolute_error(y_true, y_pred)
    }
    return metrics


def save_metrics_report(metrics: Dict[str, float]):
    """Saves the evaluation metrics to a JSON file in the artifacts/reports directory."""
    report_path = REPORTS_DIR / 'final_evaluation_metrics.json'
    
    # Convert numpy types (if any) to standard Python types for JSON serialization
    serializable_metrics = {k: float(v) for k, v in metrics.items()}
    
    with open(report_path, 'w') as f:
        json.dump(serializable_metrics, f, indent=4)
        
    logger.info(f"Final evaluation report saved to: {report_path}")


def evaluate_model():
    """
    Orchestrates the model evaluation pipeline.
    """
    logger.info("--- Starting Model Evaluation Pipeline ---")
    
    # 1. Load Artifacts and Test Data
    try:
        model, X_test, y_test_unscaled, y_scaler = load_artifacts_and_data()
    except Exception:
        return

    # 2. Make Predictions
    logger.info("Generating predictions on the test set...")
    
    # The full pipeline handles X-transformation internally.
    # Prediction output is in the SCALED Y-space.
    y_pred_scaled = model.predict(X_test)

    # 3. Inverse Transform Predictions
    # CRUCIAL: Convert the scaled prediction back to the original price units (USD).
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()

    # 4. Calculate Metrics using UNSEALED values
    metrics = calculate_metrics(y_test_unscaled, y_pred_unscaled)
    
    logger.info("\n--- FINAL MODEL PERFORMANCE ON TEST SET ---")
    for name, value in metrics.items():
        logger.info(f"{name}: {value:,.2f}") # Log with commas for readability
    logger.info("------------------------------------------")

    # 5. Save Report
    save_metrics_report(metrics)
    
    logger.info("--- Model Evaluation Complete ---")


if __name__ == "__main__":
    evaluate_model()