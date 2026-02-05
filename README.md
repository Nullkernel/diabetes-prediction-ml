# Diabetes Prediction with Machine Learning

A machine learning application that predicts diabetes using the Pima Indians Diabetes Dataset. Compares Logistic Regression and Random Forest models.

## Overview

This project implements a full-stack web application for diabetes prediction using machine learning. The application allows users to input medical parameters and receive predictions from two different ML models.

## Dataset

Pima Indians Diabetes Dataset
- Source: UCI Machine Learning Repository
- Features: 8 medical predictor variables
- Target: Binary outcome (0 = No Diabetes, 1 = Diabetes)
- Samples: 768 instances

### Features
1. Pregnancies: Number of times pregnant
2. Glucose: Plasma glucose concentration (mg/dL)
3. Blood Pressure: Diastolic blood pressure (mm Hg)
4. Skin Thickness: Triceps skin fold thickness (mm)
5. Insulin: 2-Hour serum insulin (mu U/ml)
6. BMI: Body mass index (weight in kg/(height in m)^2)
7. Diabetes Pedigree Function: Diabetes pedigree function
8. Age: Age (years)

## Models

### Logistic Regression
- Simple, interpretable baseline model
- Fast training and prediction
- Linear decision boundary

### Random Forest
- Ensemble learning method
- Better performance on non-linear patterns
- Feature importance analysis

## Model Performance

### Logistic Regression
- Accuracy: 0.7143
- Precision: 0.6087
- Recall: 0.5185
- F1-Score: 0.56
- ROC-AUC: 0.823

### Random Forest
- Accuracy: 0.7597
- Precision: 0.6809
- Recall: 0.5926
- F1-Score: 0.6337
- ROC-AUC: 0.8147

## Technology Stack

### Backend
- FastAPI: Modern Python web framework
- scikit-learn: Machine learning library
- pandas: Data manipulation
- numpy: Numerical computing

### Frontend
- React: UI library
- Tailwind CSS: Styling
- Axios: HTTP client
- Shadcn/UI: Component library

## Project Structure

```
/app/
├── backend/
│   ├── server.py              # FastAPI application with ML models
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment variables
├── frontend/
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Styles
│   │   └── components/        # UI components
│   ├── package.json           # Node dependencies
│   └── .env                   # Frontend environment
└── README.md
```

## How to Run

### Prerequisites
- Python 3.11+
- Node.js 16+
- yarn package manager

### Backend Setup
```bash
cd backend
pip install -r requirements.txt
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

### Frontend Setup
```bash
cd frontend
yarn install
yarn start
```

### Access Application
- Frontend: http://localhost:3000
- Backend API: http://localhost:8001
- API Documentation: http://localhost:8001/docs

## API Endpoints

### GET /api/models/metrics
Returns performance metrics for both models

### POST /api/predict
Makes predictions using both models

Request body:
```json
{
  "pregnancies": 6,
  "glucose": 148,
  "blood_pressure": 72,
  "skin_thickness": 35,
  "insulin": 0,
  "bmi": 33.6,
  "diabetes_pedigree_function": 0.627,
  "age": 50
}
```

## Usage

1. Open the application in your browser
2. View model performance metrics at the top
3. Enter patient medical data in the form
4. Click "Predict" to get predictions from both models
5. View results showing prediction outcomes and probabilities
6. Use "Reset" to clear the form and start over

## Important Notes

- No LLM (Large Language Model) is used for predictions
- All predictions are made using traditional machine learning algorithms
- Models are trained on the Pima Indians Diabetes Dataset
- Predictions are session-based and not stored in database
- This is an educational project for demonstration purposes
- Not intended for actual medical diagnosis

## Results

Random Forest demonstrates superior performance compared to Logistic Regression:
- 4.5% higher accuracy
- Better precision and recall
- Slightly lower ROC-AUC but better overall classification

Both models show reasonable performance for diabetes prediction, with Random Forest being the recommended model for production use.

## License

MIT License