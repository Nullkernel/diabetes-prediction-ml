from fastapi import FastAPI, APIRouter, HTTPException
from dotenv import load_dotenv
from starlette.middleware.cors import CORSMiddleware
import os
import logging
from pathlib import Path
from pydantic import BaseModel
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import warnings
warnings.filterwarnings('ignore')

ROOT_DIR = Path(__file__).parent
load_dotenv(ROOT_DIR / '.env')

# Create the main app without a prefix
app = FastAPI()

# Create a router with the /api prefix
api_router = APIRouter(prefix="/api")

# ML Model storage
models = {}
scaler = None
metrics = {}

# Define Pydantic models
class PredictionInput(BaseModel):
    pregnancies: float
    glucose: float
    blood_pressure: float
    skin_thickness: float
    insulin: float
    bmi: float
    diabetes_pedigree_function: float
    age: float

class PredictionOutput(BaseModel):
    logistic_regression: dict
    random_forest: dict

class ModelMetrics(BaseModel):
    logistic_regression: dict
    random_forest: dict

# Load and train models on startup
def load_and_train_models():
    global models, scaler, metrics
    
    # Load Pima Indians Diabetes Dataset
    url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
    column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
    
    try:
        df = pd.read_csv(url, names=column_names)
        
        # Separate features and target
        X = df.drop('Outcome', axis=1)
        y = df['Outcome']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        # Scale the features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train Logistic Regression
        lr_model = LogisticRegression(max_iter=1000, random_state=42)
        lr_model.fit(X_train_scaled, y_train)
        lr_pred = lr_model.predict(X_test_scaled)
        lr_pred_proba = lr_model.predict_proba(X_test_scaled)[:, 1]
        
        # Train Random Forest
        rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_model.fit(X_train_scaled, y_train)
        rf_pred = rf_model.predict(X_test_scaled)
        rf_pred_proba = rf_model.predict_proba(X_test_scaled)[:, 1]
        
        # Store models
        models['logistic_regression'] = lr_model
        models['random_forest'] = rf_model
        
        # Calculate and store metrics
        metrics['logistic_regression'] = {
            'accuracy': round(accuracy_score(y_test, lr_pred), 4),
            'precision': round(precision_score(y_test, lr_pred), 4),
            'recall': round(recall_score(y_test, lr_pred), 4),
            'f1_score': round(f1_score(y_test, lr_pred), 4),
            'roc_auc': round(roc_auc_score(y_test, lr_pred_proba), 4)
        }
        
        metrics['random_forest'] = {
            'accuracy': round(accuracy_score(y_test, rf_pred), 4),
            'precision': round(precision_score(y_test, rf_pred), 4),
            'recall': round(recall_score(y_test, rf_pred), 4),
            'f1_score': round(f1_score(y_test, rf_pred), 4),
            'roc_auc': round(roc_auc_score(y_test, rf_pred_proba), 4)
        }
        
        logging.info("Models trained successfully")
        logging.info(f"Logistic Regression Metrics: {metrics['logistic_regression']}")
        logging.info(f"Random Forest Metrics: {metrics['random_forest']}")
        
    except Exception as e:
        logging.error(f"Error loading/training models: {str(e)}")
        raise

@app.on_event("startup")
async def startup_event():
    load_and_train_models()

@api_router.get("/")
async def root():
    return {"message": "Diabetes Prediction API"}

@api_router.get("/models/metrics", response_model=ModelMetrics)
async def get_model_metrics():
    """Get performance metrics for both models"""
    if not metrics:
        raise HTTPException(status_code=500, detail="Models not loaded")
    return metrics

@api_router.post("/predict", response_model=PredictionOutput)
async def predict_diabetes(input_data: PredictionInput):
    """Make predictions using both models"""
    if not models or scaler is None:
        raise HTTPException(status_code=500, detail="Models not loaded")
    
    try:
        # Prepare input data
        input_array = np.array([[
            input_data.pregnancies,
            input_data.glucose,
            input_data.blood_pressure,
            input_data.skin_thickness,
            input_data.insulin,
            input_data.bmi,
            input_data.diabetes_pedigree_function,
            input_data.age
        ]])
        
        # Scale the input
        input_scaled = scaler.transform(input_array)
        
        # Get predictions from both models
        lr_pred = models['logistic_regression'].predict(input_scaled)[0]
        lr_proba = models['logistic_regression'].predict_proba(input_scaled)[0]
        
        rf_pred = models['random_forest'].predict(input_scaled)[0]
        rf_proba = models['random_forest'].predict_proba(input_scaled)[0]
        
        return {
            'logistic_regression': {
                'prediction': int(lr_pred),
                'probability_no_diabetes': round(float(lr_proba[0]), 4),
                'probability_diabetes': round(float(lr_proba[1]), 4),
                'result': 'Diabetic' if lr_pred == 1 else 'Not Diabetic'
            },
            'random_forest': {
                'prediction': int(rf_pred),
                'probability_no_diabetes': round(float(rf_proba[0]), 4),
                'probability_diabetes': round(float(rf_proba[1]), 4),
                'result': 'Diabetic' if rf_pred == 1 else 'Not Diabetic'
            }
        }
        
    except Exception as e:
        logging.error(f"Prediction error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

# Include the router in the main app
app.include_router(api_router)

app.add_middleware(
    CORSMiddleware,
    allow_credentials=True,
    allow_origins=os.environ.get('CORS_ORIGINS', '*').split(','),
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)