# src/api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import logging
from typing import Dict, Any

# Import the inference function and the input schema
from src.inference.predict import predict_price
from src.api.schemas import HouseInput

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# Initialize the FastAPI application
app = FastAPI(
    title="House Price Prediction API",
    description="A service for predicting house prices using a trained Random Forest model.",
    version="1.0"
)

# Root endpoint for health check
@app.get("/")
def read_root():
    return {"message": "House Price Prediction Service is running."}

# Prediction endpoint
@app.post("/predict", response_model=Dict[str, float])
async def predict(house_data: HouseInput):
    """
    Receives house features, generates a prediction, and returns the price.
    """
    try:
        # Convert Pydantic model to a standard dictionary expected by predict_price
        raw_data_dict = house_data.model_dump()
        
        # Call the core prediction logic
        predicted_price = predict_price(raw_data_dict)
        
        logger.info(f"Prediction successful. Price: ${predicted_price:,.2f}")
        
        # Return the result as a standard JSON dictionary
        return {"predicted_price": round(predicted_price, 2)}
        
    except Exception as e:
        logger.error(f"Prediction failed due to internal error: {e}")
        return {"error": str(e)}, 500


# --- Execution Command ---
# To run locally: uvicorn src.api.app:app --reload