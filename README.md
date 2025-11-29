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

##  Repository Structure

```text
housing_price_predictor/
├── .gitignore
├── README.md
├── run_training.py # Orchestrates the entire pipeline
├── setup.py # Enables package installation
│
├── artifacts/ # Stores all results and trained components
│ ├── figures/ # Evaluation plots (e.g., RMSE comparison)
│ ├── models/ # Final trained model pipeline (full_pipeline.joblib)
│ ├── preprocessors/ # Fitted scalers and transformers (y_scaler.joblib, etc.)
│ └── reports/ # Metric reports (evaluation_metrics.json, optimized_params.json)
│
├── data/ # Manages raw, interim, and final split data
│ ├── processed/ # X_train.csv, y_test.csv, etc.
│ └── raw/ # housing_raw.csv
│
├── notebooks/ # Jupyter notebooks for EDA and experimentation
│
└── src/ # Source Code Directory
├── api/ # FastAPI application and Pydantic schemas
│ ├── app.py # FastAPI server logic
│ └── schemas.py # Pydantic data validation model
├── config.py # Centralized configuration (paths, hyperparameters, features)
├── data/ # Data loading and structural cleaning
│ └── data_ingestion.py
├── features/ # Feature engineering and transformation logic
│ └── feature_transformer.py
└── models/ # Model training, optimization, and evaluation
├── evaluate_model.py
├── optimize_model.py
└── train_model.py
```
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
```
2. Configuration and Data
Place Data: Put your raw housing_raw.csv file into the data/raw/ directory.

Verify Configuration: Ensure that the feature lists and file paths in src/config.py match your expected column names.

🚀 Execution of the ML Pipeline
To train the model, find the optimal hyperparameters, and evaluate performance, execute the orchestrator script:

```Bash

# Must be run from the project root with the venv active

python run_training.py
This single command executes the entire complex sequence: Ingestion → Transformation → Optimization → Final Training → Evaluation.
```
🌐 Running the Prediction API
The project uses FastAPI and Uvicorn to serve the final model.

1. Start the Server
```Bash

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
```
## 🐳 Deployment via Docker

The project is containerized using a multi-stage Docker build to ensure a small, efficient, and reproducible production environment. The model is pre-trained and loaded from the container's file system.

### Local testing
```Bash
# Build the Docker image
docker build -t house-predictor . --no-cache

# Run the container (detached mode)
docker run -d -p 8000:8000 house-predictor
```

Access the API documentation at http://localhost:8000/docs.

### Deployment to Render

Push to GitHub: Ensure the final Dockerfile and all code/artifacts are committed.

Connect Render: In the Render Dashboard, create a new Web Service.

Select Docker: Choose the Docker environment, connect your GitHub repository, and deploy.

## License

This project is licensed under the MIT License.
