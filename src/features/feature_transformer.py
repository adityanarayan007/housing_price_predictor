# src/features/feature_transformer.py

import logging
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
import joblib
import os

# Import ALL necessary constants from the central config file
from src.config import (
    TRAIN_DATA_PATH,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    FINAL_TRANSFORMER_PATH,
    TEST_DATA_PATH
)

# Set up logging (inherits from basicConfig, but good practice to get instance)
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

def save_data(data: pd.DataFrame, path: str, name: str):
    """Saves a DataFrame to a specified path as a CSV file."""
    # Ensure the directory exists (safe practice)
    os.makedirs(os.path.dirname(path), exist_ok=True) 
    data.to_csv(path, index=False)
    logger.info(f"Successfully saved the {name} to: {path}")

# --- 1. Custom Feature Creation ---
def create_new_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Creates engineering features necessary for the model (e.g., binary indicators) 
    before the ColumnTransformer runs.
    """
    logger.info("Creating binary 'does_basement_exist' feature.")
    
    # Structural zero check: if basement_area_in_sqft > 0, the basement exists (1), otherwise 0.
    # We use .copy() to avoid SettingWithCopyWarning if input df is a slice
    df_copy = df.copy()
    df_copy['does_basement_exist'] = np.where(df_copy['basement_area_in_sqft'] > 0, 1, 0)
    df_copy['does_basement_exist'] = df_copy['does_basement_exist'].astype(str)
    
    # Drop the original 'basement_area_in_sqft' for the purposes of this function 
    # to avoid double creation, though it will be picked up later via NUMERICAL_FEATURES.
    # This is fine as long as we return the column needed for the transformer.
    
    return df_copy


# --- 2. Preprocessor Definition ---
def create_preprocessor() -> ColumnTransformer:
    """
    Creates the scikit-learn ColumnTransformer object based on config settings.
    This defines the full set of preprocessing steps.
    """
    logger.info("Defining numerical and categorical transformation pipelines.")
    
    # --- Numerical Pipeline: Impute -> Scale ---
    numerical_pipeline = Pipeline(steps=[
        # Since the Housing dataset can have outliers, median imputation is robust
        ('imputer', SimpleImputer(strategy='median')), 
        ('scaler', StandardScaler())
    ])
    
    # --- Categorical Pipeline: Impute -> One-Hot Encode ---
    categorical_pipeline = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        # handle_unknown='ignore' is CRUCIAL for deployment robustness
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, drop='first')) 
    ])
    
    # --- Column Transformer ---
    # Combines the pipelines and selects the columns defined in config.py
    preprocessor = ColumnTransformer(
        transformers=[
            ('numerical', numerical_pipeline, NUMERICAL_FEATURES),
            ('categorical', categorical_pipeline, CATEGORICAL_FEATURES)
        ],
        # 'remainder' is critical: 'passthrough' ensures any columns NOT listed 
        # (like any unused technical columns) are kept, but since we ensure all 
        # features are either numerical or categorical, this is often 'drop' 
        # in production, but 'passthrough' is safer for templates.
        remainder='drop',
        verbose_feature_names_out=False # NEW: Keeps output feature names clean
    )
    logger.info(f"Preprocessor defined for {len(NUMERICAL_FEATURES)} numerical and {len(CATEGORICAL_FEATURES)} categorical features.")
    return preprocessor


# --- 3. Orchestration and Saving ---
def fit_and_save_transformer():
    """
    Loads training data, creates new features, fits the ColumnTransformer, and saves the fitted object.
    """
    logger.info("--- Starting Feature Transformer Pipeline ---")
    
    # 1. Load Training Data (X_train)
    try:
        X_train = pd.read_csv(TRAIN_DATA_PATH)
        X_test = pd.read_csv(TEST_DATA_PATH)
    except FileNotFoundError:
        logger.error(f"Training data not found at {TRAIN_DATA_PATH}. Run data_ingestion first.")
        return

    # 2. CREATE NEW FEATURES
    # This adds 'does_basement_exist' using logic defined above
    X_train = create_new_features(X_train)
    X_test =  create_new_features(X_test)

    logger.info("Cleaning and standardizing categorical columns before saving.")
    
    for df in [X_train, X_test]:
        for col in CATEGORICAL_FEATURES:
            if col in df.columns:
                # Convert everything to a string type, removing float artifacts like ".0"
                df[col] = (
                    df[col].astype(str)
                    .str.replace(r'\.0$', '', regex=True) # Removes decimals
                    .str.strip()                          # Removes standard whitespace
                    .str.replace(r'[^\x00-\x7F]+', '', regex=True) # <-- Removes non-ASCII/invisible characters
                    .str.lower()                          # <-- Forces everything to lowercase
                )
            
    logger.info(f"{X_train['zipcode'].unique() = }")
    logger.info(f"{X_test['zipcode'].unique() = }")
    
    save_data(X_train, TRAIN_DATA_PATH, "engineered X_test features")
    save_data(X_test, TEST_DATA_PATH, "engineered X_test features")
   
    # 3. Instantiate and Fit Transformer
    preprocessor = create_preprocessor()
    logger.info("Fitting ColumnTransformer to training features...")
    
    # Fit the transformer only on the training set (crucial for avoiding data leakage)
    preprocessor.fit(X_train) 
    
    # 4. Save the Fitted Transformer Artifact
    joblib.dump(preprocessor, FINAL_TRANSFORMER_PATH)
    logger.info(f"Successfully saved fitted ColumnTransformer to: {FINAL_TRANSFORMER_PATH}")
    logger.info("--- Feature Transformer Complete ---")


if __name__ == "__main__":
    fit_and_save_transformer()