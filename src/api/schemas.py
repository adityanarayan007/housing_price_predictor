# src/api/schemas.py
from pydantic import BaseModel, Field
from typing import Optional, List

# Define the exact input structure for ALL 17 features
# Use the clean snake_case names and ensure correct types.
# Use Python's float/int/str/bool types, Pydantic handles validation.

class HouseInput(BaseModel):
    # Numerical Features (Examples)
    no_of_bedrooms: int = Field(..., example=2)
    no_of_bathrooms: float = Field(..., example=2.5)
    flat_area_in_sqft: float = Field(..., example=1800.0)
    lot_area_in_sqft: float = Field(..., example=9000.0)
    no_of_floors: float = Field(..., example=2.0)
    overall_grade: int = Field(..., example=8)
    age_of_house_in_years: int = Field(..., example=15)
    latitude: float = Field(..., example=47.6)
    longitude: float = Field(..., example=-122.3)
    living_area_after_renovation_in_sqft: float = Field(..., example=1800.0)
    lot_area_after_renovation_in_sqft: int = Field(..., example=8000)
    area_of_the_house_from_basement_in_sqft: int = Field(..., example=0)
    basement_area_in_sqft: int = Field(..., example=300)
    
    # Categorical Features (Must be str for consistency/OHE)
    waterfront_view: str = Field(..., example="No")
    condition_of_the_house: str = Field(..., example="Fair")
    zipcode: str = Field(..., example="98107")
    
    # Note: does_basement_exist is NOT here; it's derived internally!
    # Ensure all remaining numerical/categorical features are included.