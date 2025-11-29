# src/inference/predict.py

import logging
import joblib
import pandas as pd
from typing import Dict, Any, Tuple
import numpy as np

# Import configuration and artifact paths
from src.config import FINAL_MODEL_PATH, Y_SCALER_PATH, CATEGORICAL_FEATURES
# NOTE: Assume FINAL_MODEL_PATH points to the full scikit-learn Pipeline (X-transformer + SVR).

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


# Global variables to hold loaded artifacts (cached for deployment efficiency)
MODEL_PIPELINE = None
Y_SCALER = None


def load_artifacts(model_path: str, y_scaler_path: str) -> Tuple[Any, Any]:
    """
    Loads the full trained model pipeline (X-transformer + Estimator) 
    and the separate target scaler (Y-Scaler) from disk.
    
    The artifacts are cached in global variables to speed up subsequent predictions.
    
    Returns:
        Tuple: (fitted_model_pipeline, fitted_y_scaler)
    """
    global MODEL_PIPELINE, Y_SCALER
    
    if MODEL_PIPELINE is None or Y_SCALER is None:
        logger.info("Loading model and Y-scaler artifacts...")
        try:
            # The model is the full pipeline, which handles X-scaling/encoding
            MODEL_PIPELINE = joblib.load(model_path)
            # Y_SCALER is saved separately because it is only applied to the target
            Y_SCALER = joblib.load(y_scaler_path)
            logger.info("Artifacts loaded successfully.")
        except FileNotFoundError as e:
            logger.error(f"Failed to load required artifacts: {e}. Ensure training was run.")
            raise

    return MODEL_PIPELINE, Y_SCALER


def predict_price(raw_data: Dict[str, Any]) -> float:
    """
    Takes raw, unscaled feature data (e.g., from a UI) and returns the final house price prediction.
    
    Args:
        raw_data (Dict): A dictionary where keys are clean snake_case feature names 
                         (e.g., 'no_of_bedrooms', 'zipcode').
        
    Returns:
        float: The predicted house price in the original currency units.
    """
    # Load artifacts (will use cached versions after the first run)
    model, y_scaler = load_artifacts(FINAL_MODEL_PATH, Y_SCALER_PATH)
    
    # 1. Convert raw input dictionary to a DataFrame
    # Note: Wrap the dictionary in a list to ensure DataFrame has one row (instance).
    # The ColumnTransformer relies on seeing the columns in the correct order/schema.
    input_df = pd.DataFrame([raw_data])

    #REPLICATE FEATURE ENGINEERING LOGIC ***
    logger.info("Deriving 'does_basement_exist' feature.")
    # This logic MUST exactly match the create_new_features function in training
    input_df['does_basement_exist'] = np.where(input_df['basement_area_in_sqft'] > 0, 1, 0)
    # ***************************************************************

    #Categorical features are initially float64, in dataingestion typecasted to str type
    for col in CATEGORICAL_FEATURES:
        if col in input_df.columns:
            
            if col in ['waterfront_view', 'condition_of_the_house']:
                # **STRINGS**: Normalize case and remove space/float artifacts
                input_df[col] = input_df[col].astype(str).str.strip().str.lower()
            
            elif col in ['zipcode', 'does_basement_exist']:
                # **NUMERICS**: Must be sent as numbers (integers)
                # Ensure the number is clean (no .0 float remnant) before casting to int
                input_df[col] = pd.to_numeric(
                    input_df[col].astype(str).str.replace(r'\.0$', '', regex=True), 
                    errors='coerce' # Handle any remaining nans/errors
                ).astype('Int64') # Use Int64 for nullable integer
            else:
                # Default safety for any other unforeseen categorical feature
                input_df[col] = input_df[col].astype(str).str.strip().str.lower()
    # 2. Predict on the raw input. 
    # The full pipeline handles the X-scaling/encoding internally.
    #logger.info(f"Input columns : {input_df.columns}")
    '''first_row = input_df.iloc[0]
    for column_name, value in first_row.items():
        logger.info(f"Column '{column_name}': Value '{value}' has dtype {type(value)}")'''
    #logger.info(f"Input values : {input_df.iloc[0]}")
    logger.info("Predicting price using the full pipeline...")
    
    # Prediction will be in the SCALED Y-space
    y_pred_scaled = model.predict(input_df)
    
    # 3. Inverse Transform to Unscale the Price
    # Reshape the prediction to fit the scaler's expected input (N, 1)
    y_pred_unscaled = y_scaler.inverse_transform(y_pred_scaled.reshape(-1, 1))
    
    # Return the first (and only) prediction as a simple float
    return float(y_pred_unscaled[0][0])

'''To be removed the below code
def inspect_ohe_categories(transformer_path: str):
    """
    Loads the fitted ColumnTransformer, extracts the OneHotEncoder step, 
    and prints the categories learned for each categorical column.
    """
    try:
        # 1. Load the full ColumnTransformer (preprocessor)
        full_pipeline = joblib.load(transformer_path)
        
        # NOTE: If FINAL_MODEL_PATH points to the full Pipeline (preprocessor -> estimator), 
        # you need to extract the preprocessor first. Assuming the preprocessor is the first step:
        if hasattr(full_pipeline, 'steps'):
            # The preprocessor is the first step in the full model pipeline
            preprocessor = full_pipeline.named_steps['preprocessor']
        else:
            # Assume transformer_path points directly to the ColumnTransformer artifact
            preprocessor = full_pipeline 
            
    except FileNotFoundError:
        print(f"Error: Transformer artifact not found at {transformer_path}")
        return
    except KeyError:
        print("Error: Could not find 'preprocessor' step. Check pipeline step name.")
        return

    # 2. Find the categorical pipeline within the ColumnTransformer
    # Assuming your ColumnTransformer step for categorical features is named 'categorical'
    try:
        # Get the tuple: (name, transformer, columns) for the 'categorical' step
        categorical_step_info = next(
            (name, transformer) for name, transformer, cols in preprocessor.transformers_ 
            if name == 'categorical'
        )
        
        # The transformer is the Pipeline object for categorical features
        categorical_pipeline = categorical_step_info[1]
        
        # 3. Extract the OneHotEncoder (assuming it's the last step named 'onehot')
        ohe = categorical_pipeline.named_steps['onehot']
        
        # 4. Get the learned categories
        learned_categories = ohe.categories_
        
        # Assuming the categorical features are processed in the order of CATEGORICAL_FEATURES
        from src.config import CATEGORICAL_FEATURES
        
        print("\n--- Categories Learned by the OneHotEncoder ---")
        
        # Iterate over learned categories based on the expected feature order
        for i, col in enumerate(CATEGORICAL_FEATURES):
            if i < len(learned_categories):
                # Print the number of unique categories to spot the zip codes
                num_unique = len(learned_categories[i])
                
                print(f"\nFeature: {col} (Column {i}) | Unique Count: {num_unique}")
                
                # Print the first 10 categories to check for formatting errors
                display_list = learned_categories[i][:10]
                
                # Check for the problematic columns
                if col == 'zipcode':
                    print("  Status: ZIPCODE FORMAT CHECK IS CRITICAL!")
                elif col == 'does_basement_exist':
                    print("  Status: BASEMENT FORMAT CHECK IS CRITICAL!")
                    
                print(f"  Sample Learned Categories (First 10): {display_list}")
                
            else:
                print(f"Error: Learned categories list is shorter than expected. Missing {col}.")
                
    except Exception as e:
        print(f"Failed to access OHE or categories. Check step names ('categorical', 'onehot'). Error: {e}")

# --- EXECUTION ---
to be removed end'''

if __name__ == "__main__":
    # --- Example Usage (Replace with ALL your final snake_case feature names) ---
    # This block verifies the inference logic works, assuming artifacts are present.
    example_input = {
        'no_of_bedrooms': 2,
        'no_of_bathrooms': 2.5,
        'flat_area_in_sqft': 1800,
        'lot_area_in_sqft': 9000,
        'no_of_floors': 2,
        'waterfront_view': 'No',
        'condition_of_the_house': 'Fair',
        'overall_grade': 8,
        'area_of_the_house_from_basement_in_sqft': 0,
        'basement_area_in_sqft': 300,
        'age_of_house_in_years': 15,
        'zipcode': '98107',
        'latitude': 47.6,
        'longitude': -122.3,
        'living_area_after_renovation_in_sqft': 1800,
        'lot_area_after_renovation_in_sqft': 8000 
    }
    #from src.config import FINAL_MODEL_PATH
    
    # Run the inspection using your final model artifact path
    #inspect_ohe_categories(FINAL_MODEL_PATH)
    try:
        # Note: You need to run the full training pipeline successfully before running this.
        predicted_price = predict_price(example_input)
        logger.info(f"Test Prediction Successful. Predicted Price: ${predicted_price:,.2f}")
    except Exception:
        logger.warning("Test prediction failed. Ensure run_training.py has been executed successfully to create artifacts.")