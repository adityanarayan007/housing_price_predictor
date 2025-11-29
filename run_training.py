# run_training.py

import logging
import sys
import os

# 💡 FIX: Add the project root (src/) to the Python path
# This allows the orchestrator to correctly import modules from src/
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

# --- Import Core Execution Functions ---
# We import the main function from each sequential script
from data.data_ingestion import prepare_dataset
from features.feature_transformer import fit_and_save_transformer
from models.optimize_model import optimize_hyperparams # Used to find best params
from models.train_pipeline import train_pipeline 
from models.evaluate_model import evaluate_model # To check performance after training

# --- Setup Logging ---
logger = logging.getLogger('TrainingOrchestrator')
logger.setLevel(logging.INFO)
# Standard setup for visibility (will inherit basicConfig if run from CLI)
'''if not logger.handlers: 
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)'''


def run_full_pipeline():
    """
    Executes the complete end-to-end Machine Learning training, optimization, 
    and evaluation pipeline in the correct sequence.
    """
    logger.info("==========================================================")
    logger.info("🚀 Starting Full ML Pipeline Orchestration (Random Forest) 🚀")
    logger.info("==========================================================")

    try:
        # --- 1. Data Ingestion and Structural Cleaning ---
        logger.info("\n--- STAGE 1/5: Data Ingestion and Split ---")
        prepare_dataset()
        logger.info("✅ Data Ingestion Complete.")

        # --- 2. Feature Transformation (Fitting Preprocessor and Y-Scaler) ---
        logger.info("\n--- STAGE 2/5: Feature Transformation and Artifacts Fit ---")
        fit_and_save_transformer() # Fits X-transformer
        # NOTE: Y-Scaler is fitted within optimize_model.py, so we proceed to next stage.
        logger.info("✅ X-Transformer Fitted and Saved.")

        # --- 3. Hyperparameter Optimization ---
        logger.info("\n--- STAGE 3/5: Hyperparameter Optimization (RandomizedSearchCV) ---")
        optimize_hyperparams() # Finds and saves best parameters to artifacts/reports/optimized_params.json
        logger.info("✅ Optimization Complete. Best parameters saved.")
        
        # --- 4. Final Model Training ---
        logger.info("\n--- STAGE 4/5: Final Model Training ---")
        train_pipeline() # Loads optimized params and trains the final model pipeline
        logger.info("✅ Final Model Training and Saving Complete.")
        
        # --- 5. Model Evaluation ---
        logger.info("\n--- STAGE 5/5: Model Evaluation ---")
        evaluate_model() # Evaluates the final model on the test set
        logger.info("✅ Model Evaluation Complete.")

        logger.info("\n==========================================================")
        logger.info("🎉 Full Pipeline Executed Successfully! Artifacts Ready. 🎉")
        logger.info("==========================================================")
        
    except Exception as e:
        logger.error(f"\n❌ Pipeline failed at a critical stage! Error: {e}", exc_info=True)
        # Exit with a non-zero status code (standard MLOps practice for failure)
        sys.exit(1) 


if __name__ == "__main__":
    run_full_pipeline()