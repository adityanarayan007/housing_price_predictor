# src/config.py

import os
from pathlib import Path
from typing import List, Dict, Any

# ==============================================================================
# 1. CORE PROJECT SETUP AND PATHS 
# ==============================================================================

# Dynamically find the project root (assumes config.py is inside src/)
# ROOT_DIR points to the top-level directory (e.g., parentFolderName/)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Define main structural directories
DATA_DIR = ROOT_DIR / 'data'
ARTIFACTS_DIR = ROOT_DIR / 'artifacts'

# Define subdirectories for organization
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'
MODELS_DIR = ARTIFACTS_DIR / 'models'
PREPROCESSORS_DIR = ARTIFACTS_DIR / 'preprocessors'
REPORTS_DIR = ARTIFACTS_DIR / 'reports'

# Create directories if they don't exist (MLOps best practice for reliability)
for d in [MODELS_DIR, PREPROCESSORS_DIR, REPORTS_DIR, PROCESSED_DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==============================================================================
# 2. FILE PATHS (Project Specific Naming)
# ==============================================================================

# **UPDATED** to housing dataset file name.
RAW_FILE_NAME = "housing_raw.csv" 
MODEL_PIPELINE_NAME = "full_pipeline.joblib"
TRANSFORMER_NAME = "feature_transformer.joblib"

# Full Paths (used by data_loader and model_trainer)
RAW_DATA_PATH = str(RAW_DATA_DIR / RAW_FILE_NAME)
TRAIN_DATA_PATH = str(PROCESSED_DATA_DIR / "X_train.csv")
TEST_DATA_PATH = str(PROCESSED_DATA_DIR / "X_test.csv")
TRAIN_TARGET_PATH = str(PROCESSED_DATA_DIR / "y_train.csv")
TEST_TARGET_PATH = str(PROCESSED_DATA_DIR / "y_test.csv")

FINAL_MODEL_PATH = str(MODELS_DIR / MODEL_PIPELINE_NAME)
FINAL_TRANSFORMER_PATH = str(PREPROCESSORS_DIR / TRANSFORMER_NAME)


# ==============================================================================
# 3. TRAINING & MODEL HYPERPARAMETERS (Random Forest Regressor)
# ==============================================================================

TEST_SIZE: float = 0.2
RANDOM_STATE: int = 42

# selected model from your experimentation
MODEL_TYPE: str = 'RandomForestRegressor' 

# starting hyperparameters for Random Forest
DEFAULT_HYPERPARAMS: Dict[str, Any] = {
    'n_estimators': 100, 
    'max_depth': 15,
    'min_samples_leaf': 5,
    'random_state': 42
}

# --- Y-SCALER PATH (Needed since target has been scaled) ---
Y_SCALER_PATH = str(PREPROCESSORS_DIR / 'y_scaler.joblib')


# ==============================================================================
# 4. FEATURE DEFINITIONS (FINALIZED SNAKE_CASE LISTS)
# ==============================================================================
# These names are the clean, snake_case names *after* renaming in data_ingestion.py.

TARGET_COLUMN: str = 'sale_price' 

# ------------------------------------------------------------------------------
# 4A. CATEGORICAL FEATURES (Need One-Hot Encoding/Imputation)
# ------------------------------------------------------------------------------
CATEGORICAL_FEATURES: List[str] = [
    'waterfront_view',    
    'condition_of_the_house',
    'zipcode',
    'does_basement_exist'
]

# ------------------------------------------------------------------------------
# 4B. NUMERICAL FEATURES (Need Scaling/Imputation)
# ------------------------------------------------------------------------------
NUMERICAL_FEATURES: List[str] = [
    'no_of_bedrooms', 'no_of_bathrooms', 'flat_area_in_sqft', 
    'lot_area_in_sqft', 'no_of_floors', 'overall_grade', 
    'area_of_the_house_from_basement_in_sqft', 'basement_area_in_sqft',
    'age_of_house_in_years', 'latitude', 'longitude', 
    'living_area_after_renovation_in_sqft', 'lot_area_after_renovation_in_sqft'
]

# ------------------------------------------------------------------------------
# 4C. FEATURES TO DROP
# ------------------------------------------------------------------------------
# columns should be dropped during ingestion.
DROP_FEATURES: List[str] = [
    'id' ,
    'date_house_was_sold',
    'renovated_year',
    "no_of_times_visited"
]