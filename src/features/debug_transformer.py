# src/features/debug_transformer.py

import joblib
import pandas as pd
import numpy as np
import os
import sys

# Add project root to path to ensure imports work when running this script
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

from src.config import TRAIN_DATA_PATH, TEST_DATA_PATH, FINAL_TRANSFORMER_PATH, CATEGORICAL_FEATURES, DATA_DIR
from src.features.feature_transformer import create_new_features 
# Note: Ensure create_new_features is available/imported correctly here if not in __init__.py

# --- 1. Data Preparation Utility ---

def prepare_data_for_transform(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies feature engineering (create_new_features) and the necessary 
    type standardization/cleaning required before the ColumnTransformer runs.
    """
    print("Applying create_new_features...")
    df = create_new_features(df)
    
    # CRITICAL DEBUGGING STEP: Replicate the robust cleaning 
    # that should be done in feature_transformer.py before saving.
    print("Applying robust categorical string cleanup...")
    for col in CATEGORICAL_FEATURES:
        if col in df.columns:
            # Cast to string, remove the trailing ".0" (from float conversion), and strip spaces
            df[col] = df[col].astype(str).str.replace(r'\.0$', '', regex=True).str.strip()
    
    return df

# --- 2. Main Debugging Function ---

def save_transformed_data():
    """
    Loads X_train and X_test, prepares them, transforms them using the fitted 
    preprocessor, and saves the final NumPy arrays to the data/final folder for inspection.
    """
    print("--- Starting Transformed Data Save for Debugging ---")
    
    # Define output paths: Using PROCESSED_DATA_DIR (which points to data/processed) 
    # and creating a final sub-folder within it for clarity.
    OUTPUT_DIR = os.path.join(DATA_DIR, 'final') 
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Adjusting file names based on user request to be clear they are transformed
    TRAIN_OUT_PATH = os.path.join(OUTPUT_DIR, 'final_X_train_transformed.csv')
    TEST_OUT_PATH = os.path.join(OUTPUT_DIR, 'final_X_test_transformed.csv')

    try:
        # 1. Load Data and Artifact
        X_train = pd.read_csv(TRAIN_DATA_PATH)
        X_test = pd.read_csv(TEST_DATA_PATH)
        # Assuming FINAL_TRANSFORMER_PATH holds the fitted ColumnTransformer
        preprocessor = joblib.load(FINAL_TRANSFORMER_PATH)
    except FileNotFoundError as e:
        print(f"Error: Missing file—ensure data ingestion and transformer fitting are complete. {e}")
        return

    # 2. Apply Preparation (Engineering + Cleaning)
    X_train_prepared = prepare_data_for_transform(X_train)
    X_test_prepared = prepare_data_for_transform(X_test)
    
    # 3. Transform Data using the Fitted Preprocessor
    print("Applying fitted ColumnTransformer...")
    # NOTE: This is where the 'unknown categories' warning would fire if the input data is bad
    print("Tranforming X_train...")
    X_train_transformed = preprocessor.transform(X_train_prepared)

    print("Transforming X_test")
    X_test_transformed = preprocessor.transform(X_test_prepared)
    
    print(X_train['zipcode'].unique())
    print(X_test['zipcode'].unique())
    # 4. Save Transformed Arrays
    np.savetxt(TRAIN_OUT_PATH, X_train_transformed, delimiter=",")
    np.savetxt(TEST_OUT_PATH, X_test_transformed, delimiter=",")
    
    print("\n✅ Transformed data saved successfully:")
    print(f"   Train Shape: {X_train_transformed.shape}")
    print(f"   Saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    save_transformed_data()