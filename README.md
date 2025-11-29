# Housing Price Predictor

A complete end-to-end Machine Learning pipeline for predicting housing prices using structured tabular data.  
This project includes data ingestion, preprocessing, feature engineering, model training, evaluation, and artifact generation.

---

## Features

- End-to-end modular ML pipeline  
- Data ingestion module  
- Data transformation (encoding + scaling)  
- Model training and evaluation  
- Saves trained model and preprocessor  
- Clean, industry-standard project structure  
- Suitable for deployment (API / Streamlit)  

---

## Project Structure

housing_price_predictor/
│
├── artifacts/ # Trained model, transformers, metadata
├── data/ # Raw / interim / processed data
├── notebooks/ # EDA and experimentation
├── src/
│ ├── data/ # ingestion
│ ├── features/ # training & prediction pipelines
│ ├── models/ # helper functions
│ ├── inference/ # training & prediction pipelines
│ ├── app/ # helper functions
│ ├── exception.py
│ ├── logger.py
│ └── ...
│
├── run_training.py # Runs the complete pipeline
├── requirements.txt # Dependencies
├── setup.py # Packaging configuration
└── README.md # Documentation


---

## ⚙️ Setup and Installation

### 1. Environment and Dependencies

Clone the repository and set up your isolated environment.

```bash
# 1. Clone the repository and navigate into the project folder
git clone YOUR_REPO_URL
cd {{cookiecutter.project_slug}}

# 2. Create and activate environment
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate

# 3. Install Git LFS (Required for model artifact download)
git lfs install
git lfs pull # Download the full_pipeline.joblib file

# 4. Install ALL dependencies (production + development tools)
pip install -r requirements-dev.txt
2. Configuration and Data
Place Data: Put your raw housing_raw.csv file into the data/raw/ directory.

Verify Configuration: Ensure that the feature lists and file paths in src/config.py match your expected column names.

🚀 Execution of the ML Pipeline
To train the model, find the optimal hyperparameters, and evaluate performance, execute the orchestrator script:

Bash

# Must be run from the project root with the venv active

python run_training.py
This single command executes the entire complex sequence: Ingestion → Transformation → Optimization → Final Training → Evaluation.

🌐 Running the Prediction API
The project uses FastAPI and Uvicorn to serve the final model.

1. Start the Server
Bash

# Ensure venv is active
uvicorn src.api.app:app --reload
The service will start, usually on http://127.0.0.1:8000.

2. Testing and Prediction
API Documentation: Access the interactive testing interface at http://127.0.0.1:8000/docs (Swagger UI).

Prediction: Use the Swagger UI to submit a JSON request containing all 17 raw, unengineered features to the /predict endpoint.

Example Input Structure (JSON Body):

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


The API will return the predicted house price in USD.


License

This project is licensed under the MIT License.
