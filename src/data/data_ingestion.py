# src/data/data_ingestion.py

import pandas as pd
import os
import logging
from sklearn.model_selection import train_test_split 
from typing import Tuple, Dict, Any, List
import re # Needed for robust column renaming

#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
# Import ALL configurations and paths
from src.config import (
    RAW_DATA_PATH,
    TRAIN_DATA_PATH,
    TEST_DATA_PATH,
    TRAIN_TARGET_PATH,
    TEST_TARGET_PATH,
    TARGET_COLUMN, 
    DROP_FEATURES,      
    TEST_SIZE,
    RANDOM_STATE,
    CATEGORICAL_FEATURES
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

# --- 1. Structural Cleaning Functions ---

def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """
    Renames all columns to be lowercase and snake_case, replacing spaces/special chars.
    This ensures consistency throughout the pipeline.
    """
    logger.info("Starting column name cleaning (to snake_case).")
    new_columns = []
    
    for col in df.columns:
        # Convert to lowercase
        name = col.lower()
        # Replace spaces, parentheses, hyphens, and slashes with a single underscore
        name = re.sub(r'[\s()\-\/]+', '_', name) 
        # Clean up leading/trailing underscores and double underscores
        name = name.strip('_').replace('__', '_') 
        new_columns.append(name)
    
    df.columns = new_columns
    logger.info("Column names successfully cleaned.")
    logger.info(f"{df.columns}")
    logger.info(f"Shape : {df.shape}")
    return df


def load_raw_data(path: str) -> pd.DataFrame:
    """Loads the raw dataset from the specified path."""
    logger.info(f"Loading raw data from: {path}")
    try:
        # Assuming standard CSV format (sep=',') for generic template
        df = pd.read_csv(path, sep=",") 
        logger.info(f"Raw data successfully loaded. Shape: {df.shape}")
        return df
    except FileNotFoundError:
        logger.error(f"Error: Raw data file not found at {path}")
        raise


def apply_structural_cleaning(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies column renaming and drops unnecessary/low-signal features.
    """
    # 1. RENAME COLUMNS (MUST COME FIRST)
    cleaned_df = clean_column_names(raw_df)

    #dropping rows with nan value for zipcode
    initial_shape = cleaned_df.shape[0]
    cleaned_df.dropna(subset=['zipcode'], inplace=True)
    rows_dropped = initial_shape - cleaned_df.shape[0]
    logger.info(f"Dropped {rows_dropped} rows with missing 'zipcode'.")


    logger.info("Casting potential numeric-categorical columns to string type.")
    # Force 'zipcode' (and any other categorical that might be float) to string
    original_categorical_features = [
        col for col in CATEGORICAL_FEATURES 
        if col != 'does_basement_exist'
    ]
    
    # Force original categorical features to string
    for col in original_categorical_features:
        if col in cleaned_df.columns:
            cleaned_df[col] = cleaned_df[col].astype(str)
    
    # 2. DROP UNNECESSARY COLUMNS
    logger.info(f"Dropping unnecessary columns: {DROP_FEATURES}")
    # 'errors=ignore' prevents the script from crashing if a column is missing
    df_filtered = cleaned_df.drop(columns=DROP_FEATURES, errors='ignore') 
    
    logger.info(f"Data filtered. New shape: {df_filtered.shape}")
    return df_filtered


# --- 2. Splitting and Saving Functions ---

def split_data(
    X: pd.DataFrame, 
    y: pd.Series, 
    test_size: float, 
    random_state: int
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Splits the features (X) and target (y) into training and testing sets."""
    logger.info(f"Splitting X and y: Test size = {test_size:.0%}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, 
        y, 
        test_size=test_size, 
        random_state=random_state,
        # Stratify is usually for classification, but harmless for continuous target. 
        # Leaving it out here to be purely regression-focused.
    )
    logger.info(f"Split complete. X_train shape: {X_train.shape}")
    return X_train, X_test, y_train, y_test


def save_data(data: pd.DataFrame | pd.Series, path: str, name: str):
    """Saves a DataFrame or Series to a specified path as a CSV file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    
    # Convert Series to DataFrame for consistent saving with index=False
    if isinstance(data, pd.Series):
        data = data.to_frame() 
        
    data.to_csv(path, index=False)
    logger.info(f"Successfully saved the {name} to: {path}")


# --- 3. Main Orchestration Function ---

def prepare_dataset():
    """
    Runs the full dataset preparation pipeline: load, clean, separate X/y, split, and save.
    """
    logger.info("--- Starting Data Ingestion Pipeline ---")
    
    # 1. Load Data
    raw_df = load_raw_data(RAW_DATA_PATH)
    
    # 2. Apply Structural Cleaning (Rename and Drop)
    cleaned_df = apply_structural_cleaning(raw_df)
    
    # 3. Separate Features (X) and Target (y)
    if TARGET_COLUMN not in cleaned_df.columns:
        logger.error(f"Target column '{TARGET_COLUMN}' not found. Check config.py and renaming logic.")
        return
        
    X = cleaned_df.drop(columns=[TARGET_COLUMN])
    y = cleaned_df[TARGET_COLUMN]
    logger.info("Features and target successfully separated.")
    
    # 4. Split Data
    X_train, X_test, y_train, y_test = split_data(
        X=X,
        y=y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )
    
    # 5. Save Data (Four files for clean processing)
    save_data(X_train, TRAIN_DATA_PATH, "X_train features")
    save_data(X_test, TEST_DATA_PATH, "X_test features")
    save_data(y_train, TRAIN_TARGET_PATH, "y_train target")
    save_data(y_test, TEST_TARGET_PATH, "y_test target")
    
    logger.info("--- Data Ingestion Complete ---")


if __name__ == "__main__":
    prepare_dataset()