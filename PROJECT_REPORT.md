# Diabetes Prediction with Machine Learning - Project Report

## Table of Contents
1. Project Overview
2. System Architecture
3. Dataset Description
4. Data Preprocessing
5. Model Implementation
6. Model Comparison
7. Evaluation Metrics
8. User Interface
9. Results and Screenshots
10. Conclusion
11. Steps to Run

---

## 1. Project Overview

This project implements a full-stack web application for diabetes prediction using machine learning. The system uses the Pima Indians Diabetes Dataset and compares two machine learning algorithms: Logistic Regression and Random Forest.

### Objectives
- Implement and compare multiple ML models for diabetes prediction
- Create an intuitive web interface for making predictions
- Display model performance metrics for transparency
- Provide probability estimates along with predictions

### Key Features
- Single prediction form with 8 medical input parameters
- Real-time predictions from two ML models
- Model performance comparison dashboard
- Clean, minimal user interface
- No database storage (session-based predictions)

---

## 2. System Architecture

### Architecture Diagram
```
┌─────────────────┐
│   Frontend      │
│   (React)       │
│  Port: 3000     │
└────────┬────────┘
         │
         │ HTTP Requests
         │ (axios)
         │
┌────────┴────────┐
│   Backend       │
│   (FastAPI)     │
│  Port: 8001     │
└────────┬────────┘
         │
         │
    ┌────┴────┐
    │   ML      │
    │  Models   │
    │          │
    │  - LR     │
    │  - RF     │
    └─────────┘
```

### Technology Stack

#### Backend
- **Framework**: FastAPI 0.110.1
- **ML Library**: scikit-learn 1.6.1
- **Data Processing**: pandas 2.2.0, numpy 1.26.0
- **Server**: Uvicorn

#### Frontend
- **Framework**: React 19.0.0
- **Styling**: Tailwind CSS 3.4.17
- **UI Components**: Shadcn/UI (Radix UI)
- **HTTP Client**: Axios 1.8.4
- **Routing**: React Router DOM 7.5.1

#### Development Environment
- **Containerization**: Kubernetes
- **Process Manager**: Supervisor
- **Hot Reload**: Enabled for both frontend and backend

---

## 3. Dataset Description

### Pima Indians Diabetes Dataset

**Source**: UCI Machine Learning Repository

**Dataset URL**: https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv

**Description**: The dataset contains medical information about Pima Indian women aged 21 years or older. The target variable indicates whether the patient developed diabetes within 5 years.

**Total Samples**: 768 instances

**Features (8)**:

1. **Pregnancies**: Number of times pregnant (integer)
2. **Glucose**: Plasma glucose concentration at 2 hours in oral glucose tolerance test (mg/dL)
3. **BloodPressure**: Diastolic blood pressure (mm Hg)
4. **SkinThickness**: Triceps skin fold thickness (mm)
5. **Insulin**: 2-Hour serum insulin (μU/mL)
6. **BMI**: Body mass index (weight in kg/(height in m)^2)
7. **DiabetesPedigreeFunction**: A function that scores likelihood of diabetes based on family history
8. **Age**: Age in years (integer)

**Target Variable**: 
- **Outcome**: Binary classification (0 = No Diabetes, 1 = Diabetes)

**Class Distribution**:
- Class 0 (No Diabetes): 500 instances (65%)
- Class 1 (Diabetes): 268 instances (35%)

---

## 4. Data Preprocessing

### Steps Implemented

#### 1. Data Loading
```python
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 
                'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
df = pd.read_csv(url, names=column_names)
```

#### 2. Feature-Target Separation
```python
X = df.drop('Outcome', axis=1)
y = df['Outcome']
```

#### 3. Train-Test Split
- **Test Size**: 20% (154 samples)
- **Train Size**: 80% (614 samples)
- **Stratification**: Yes (maintains class distribution)
- **Random State**: 42 (for reproducibility)

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
```

#### 4. Feature Scaling
- **Method**: StandardScaler (Z-score normalization)
- **Formula**: z = (x - μ) / σ
- **Reason**: Required for Logistic Regression; improves Random Forest performance

```python
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
```

**Important**: The scaler is fit only on training data to prevent data leakage.

---

## 5. Model Implementation

### Model 1: Logistic Regression

**Purpose**: Baseline model for interpretability and comparison

**Configuration**:
```python
lr_model = LogisticRegression(max_iter=1000, random_state=42)
```

**Parameters**:
- max_iter: 1000 (ensures convergence)
- random_state: 42 (reproducibility)

**Training**:
```python
lr_model.fit(X_train_scaled, y_train)
```

**Characteristics**:
- Linear decision boundary
- Fast training and prediction
- Interpretable coefficients
- Works well with scaled features

---

### Model 2: Random Forest Classifier

**Purpose**: Ensemble model for improved performance

**Configuration**:
```python
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
```

**Parameters**:
- n_estimators: 100 (number of decision trees)
- random_state: 42 (reproducibility)

**Training**:
```python
rf_model.fit(X_train_scaled, y_train)
```

**Characteristics**:
- Non-linear decision boundaries
- Handles feature interactions
- Resistant to overfitting
- Provides feature importance

---

## 6. Model Comparison

### Performance Metrics Table

| Metric | Logistic Regression | Random Forest | Winner |
|--------|---------------------|---------------|--------|
| **Accuracy** | 0.7143 | **0.7597** | RF (+4.5%) |
| **Precision** | 0.6087 | **0.6809** | RF (+7.2%) |
| **Recall** | 0.5185 | **0.5926** | RF (+7.4%) |
| **F1-Score** | 0.56 | **0.6337** | RF (+7.4%) |
| **ROC-AUC** | **0.823** | 0.8147 | LR (+0.8%) |

### Analysis

#### Logistic Regression
**Strengths**:
- Higher ROC-AUC score (0.823)
- Faster predictions
- More interpretable
- Simpler model

**Weaknesses**:
- Lower accuracy (71.43%)
- Lower precision (60.87%)
- Lower recall (51.85%)

#### Random Forest
**Strengths**:
- Best overall accuracy (75.97%)
- Best precision (68.09%)
- Best recall (59.26%)
- Best F1-score (0.6337)
- Captures non-linear patterns

**Weaknesses**:
- Slightly lower ROC-AUC
- More complex model
- Slower predictions

### Recommendation
**Winner**: Random Forest

Random Forest is the recommended model for production use due to:
1. 4.5% higher accuracy
2. Better balance between precision and recall
3. Superior F1-score indicating better overall performance
4. Minimal ROC-AUC trade-off (0.8%)

---

## 7. Evaluation Metrics

### Metrics Explanation

#### 1. Accuracy
**Formula**: (TP + TN) / (TP + TN + FP + FN)

**Meaning**: Proportion of correct predictions

**Interpretation**:
- LR: 71.43% of predictions are correct
- RF: 75.97% of predictions are correct

#### 2. Precision
**Formula**: TP / (TP + FP)

**Meaning**: Of all positive predictions, how many were actually positive?

**Interpretation**:
- LR: 60.87% of diabetes predictions are correct
- RF: 68.09% of diabetes predictions are correct

#### 3. Recall (Sensitivity)
**Formula**: TP / (TP + FN)

**Meaning**: Of all actual positives, how many were correctly identified?

**Interpretation**:
- LR: Detects 51.85% of actual diabetes cases
- RF: Detects 59.26% of actual diabetes cases

#### 4. F1-Score
**Formula**: 2 * (Precision * Recall) / (Precision + Recall)

**Meaning**: Harmonic mean of precision and recall

**Interpretation**:
- LR: 0.56 balanced performance
- RF: 0.6337 better balanced performance

#### 5. ROC-AUC
**Meaning**: Area Under the Receiver Operating Characteristic Curve

**Interpretation**:
- LR: 0.823 excellent discrimination ability
- RF: 0.8147 excellent discrimination ability
- Closer to 1.0 is better

---

## 8. User Interface

### Design Principles
- **Minimal and Simple**: Clean layout as requested
- **Easy to Read**: Clear labels and organized sections
- **No Unnecessary Visuals**: Focus on functionality
- **Responsive**: Works on different screen sizes

### UI Components

#### 1. Header Section
- Application title
- Subtitle explaining purpose

#### 2. Model Performance Metrics Card
- Side-by-side comparison of both models
- All 5 metrics displayed for each model
- Loaded on page load

#### 3. Input Form Card
- 8 input fields for medical parameters
- Clear labels with units
- Input validation
- Predict and Reset buttons

#### 4. Prediction Results Card
- Displays results from both models
- Shows prediction outcome (Diabetic/Not Diabetic)
- Color-coded results (red for diabetic, green for not diabetic)
- Probability percentages for both classes
- Separated by model

#### 5. Error Handling
- Displays error messages if prediction fails
- Form validation for required fields

### Color Scheme
- Background: Light gray (bg-gray-50)
- Cards: White with subtle shadows
- Text: Dark gray for readability
- Positive results: Green
- Negative results: Red
- Buttons: Black with white text

---

## 9. Results and Screenshots

### Screenshot 1: Initial Page with Model Metrics
[Insert screenshot showing the main page with model performance metrics displayed at the top]

**Features Visible**:
- Model Performance Metrics card
- Logistic Regression metrics (Accuracy, Precision, Recall, F1-Score, ROC-AUC)
- Random Forest metrics
- Empty prediction form

### Screenshot 2: Form Filled with Sample Data
[Insert screenshot showing the form filled with patient data]

**Sample Data Used**:
- Pregnancies: 6
- Glucose: 148 mg/dL
- Blood Pressure: 72 mm Hg
- Skin Thickness: 35 mm
- Insulin: 0 μU/mL
- BMI: 33.6
- Diabetes Pedigree Function: 0.627
- Age: 50 years

### Screenshot 3: Prediction Results
[Insert screenshot showing prediction results from both models]

**Results**:

**Logistic Regression**:
- Result: Diabetic (red)
- Probability (No Diabetes): 26.86%
- Probability (Diabetes): 73.14%

**Random Forest**:
- Result: Diabetic (red)
- Probability (No Diabetes): 17.00%
- Probability (Diabetes): 83.00%

**Analysis**: Both models correctly predict diabetes for this high-risk profile (high glucose, high BMI, older age, family history).

---

## 10. Conclusion

### Project Summary

This project successfully implements a diabetes prediction system using machine learning with the following achievements:

1. **Dataset**: Successfully integrated Pima Indians Diabetes Dataset
2. **Models**: Implemented and compared Logistic Regression and Random Forest
3. **Performance**: Achieved 75.97% accuracy with Random Forest
4. **Interface**: Created a minimal, functional web interface
5. **Features**: Real-time predictions with probability estimates

### Key Findings

1. **Random Forest Superior Performance**: RF outperforms LR in most metrics
2. **Trade-offs**: Slight ROC-AUC advantage for LR vs better overall metrics for RF
3. **Practical Application**: Both models show reasonable performance for educational purposes
4. **User Experience**: Clean interface makes predictions accessible

### Important Disclaimers

1. **No LLM Used**: All predictions are made using traditional ML algorithms (Logistic Regression and Random Forest)
2. **Educational Purpose**: This project is for demonstration and learning
3. **Not Medical Advice**: Should not be used for actual medical diagnosis
4. **No Data Storage**: Predictions are session-based only
5. **Model Limitations**: Performance limited by dataset size and feature engineering

### Future Enhancements

Potential improvements for the project:

1. **Feature Engineering**: Add derived features, handle missing values
2. **Additional Models**: SVM, XGBoost, Neural Networks
3. **Cross-Validation**: Implement k-fold cross-validation
4. **Feature Importance**: Visualize which features matter most
5. **Batch Predictions**: Allow CSV upload for multiple predictions
6. **History Tracking**: Optional database storage for tracking predictions
7. **Model Explainability**: Add SHAP or LIME for prediction explanations
8. **Mobile Optimization**: Better responsive design for mobile devices

---

## 11. Steps to Run the Project

### Prerequisites

**Software Requirements**:
- Python 3.11 or higher
- Node.js 16 or higher
- yarn package manager
- Git (for cloning repository)

### Installation Steps

#### Step 1: Clone Repository
```bash
git clone <repository-url>
cd <project-directory>
```

#### Step 2: Backend Setup
```bash
# Navigate to backend directory
cd backend

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

#### Step 3: Configure Backend Environment
Create `.env` file in backend directory:
```
CORS_ORIGINS=*
```

#### Step 4: Start Backend Server
```bash
# From backend directory
uvicorn server:app --host 0.0.0.0 --port 8001 --reload
```

Backend will be available at: http://localhost:8001

#### Step 5: Frontend Setup
```bash
# Open new terminal
# Navigate to frontend directory
cd frontend

# Install dependencies
yarn install
```

#### Step 6: Configure Frontend Environment
Create `.env` file in frontend directory:
```
REACT_APP_BACKEND_URL=http://localhost:8001
```

#### Step 7: Start Frontend Server
```bash
# From frontend directory
yarn start
```

Frontend will open automatically at: http://localhost:3000

### Verification Steps

#### 1. Check Backend
- Visit http://localhost:8001/docs
- You should see FastAPI Swagger documentation
- Test the `/api/models/metrics` endpoint

#### 2. Check Frontend
- Visit http://localhost:3000
- You should see the Diabetes Prediction System
- Model metrics should be displayed at the top

#### 3. Test Prediction
- Fill in the form with sample data
- Click "Predict"
- Results should appear on the right side

### Troubleshooting

**Backend Issues**:
- If models fail to load, check internet connection (dataset is fetched from URL)
- Ensure all dependencies are installed correctly
- Check backend logs for error messages

**Frontend Issues**:
- Clear browser cache and reload
- Ensure backend is running before starting frontend
- Check browser console for errors
- Verify REACT_APP_BACKEND_URL is set correctly

**CORS Issues**:
- Ensure CORS_ORIGINS=* in backend .env
- Restart backend server after environment changes

### API Testing with cURL

**Get Model Metrics**:
```bash
curl http://localhost:8001/api/models/metrics
```

**Make Prediction**:
```bash
curl -X POST http://localhost:8001/api/predict \
  -H "Content-Type: application/json" \
  -d '{
    "pregnancies": 6,
    "glucose": 148,
    "blood_pressure": 72,
    "skin_thickness": 35,
    "insulin": 0,
    "bmi": 33.6,
    "diabetes_pedigree_function": 0.627,
    "age": 50
  }'
```

---

## Appendix

### A. Dataset Statistics

**Feature Statistics** (training set):
```
Pregnancies: mean=3.8, std=3.4, range=[0, 17]
Glucose: mean=120.9, std=32.0, range=[0, 199]
BloodPressure: mean=69.1, std=19.4, range=[0, 122]
SkinThickness: mean=20.5, std=16.0, range=[0, 99]
Insulin: mean=79.8, std=115.2, range=[0, 846]
BMI: mean=32.0, std=7.9, range=[0, 67.1]
DiabetesPedigreeFunction: mean=0.47, std=0.33, range=[0.08, 2.42]
Age: mean=33.2, std=11.8, range=[21, 81]
```

### B. Technology Versions

**Backend**:
- Python: 3.11
- FastAPI: 0.110.1
- scikit-learn: 1.6.1
- pandas: 2.2.0
- numpy: 1.26.0
- uvicorn: 0.25.0

**Frontend**:
- React: 19.0.0
- Tailwind CSS: 3.4.17
- Axios: 1.8.4
- React Router DOM: 7.5.1

### C. File Structure

```
/app/
├── backend/
│   ├── server.py              # Main FastAPI application
│   ├── requirements.txt       # Python dependencies
│   └── .env                   # Environment configuration
├── frontend/
│   ├── public/                # Static assets
│   ├── src/
│   │   ├── App.js             # Main React component
│   │   ├── App.css            # Component styles
│   │   ├── index.js           # Entry point
│   │   ├── index.css          # Global styles
│   │   └── components/        # UI components
│   │       └── ui/            # Shadcn/UI components
│   ├── package.json           # Node.js dependencies
│   ├── tailwind.config.js     # Tailwind configuration
│   ├── postcss.config.js      # PostCSS configuration
│   └── .env                   # Frontend environment
└── README.md              # Project documentation
```

### D. References

1. Pima Indians Diabetes Dataset: UCI Machine Learning Repository
2. scikit-learn Documentation: https://scikit-learn.org/
3. FastAPI Documentation: https://fastapi.tiangolo.com/
4. React Documentation: https://react.dev/
5. Tailwind CSS Documentation: https://tailwindcss.com/

---

**End of Project Report**