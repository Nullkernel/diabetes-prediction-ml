import { useState, useEffect } from "react";
import "@/App.css";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Label } from "@/components/ui/label";
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from "@/components/ui/card";
import { Alert, AlertDescription } from "@/components/ui/alert";
import { Separator } from "@/components/ui/separator";

const BACKEND_URL = process.env.REACT_APP_BACKEND_URL;
const API = `${BACKEND_URL}/api`;

function App() {
  const [formData, setFormData] = useState({
    pregnancies: '',
    glucose: '',
    blood_pressure: '',
    skin_thickness: '',
    insulin: '',
    bmi: '',
    diabetes_pedigree_function: '',
    age: ''
  });
  
  const [metrics, setMetrics] = useState(null);
  const [prediction, setPrediction] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchMetrics();
  }, []);

  const fetchMetrics = async () => {
    try {
      const response = await axios.get(`${API}/models/metrics`);
      setMetrics(response.data);
    } catch (e) {
      console.error('Error fetching metrics:', e);
      setError('Failed to load model metrics');
    }
  };

  const handleInputChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setPrediction(null);

    try {
      const response = await axios.post(`${API}/predict`, {
        pregnancies: parseFloat(formData.pregnancies),
        glucose: parseFloat(formData.glucose),
        blood_pressure: parseFloat(formData.blood_pressure),
        skin_thickness: parseFloat(formData.skin_thickness),
        insulin: parseFloat(formData.insulin),
        bmi: parseFloat(formData.bmi),
        diabetes_pedigree_function: parseFloat(formData.diabetes_pedigree_function),
        age: parseFloat(formData.age)
      });
      setPrediction(response.data);
    } catch (e) {
      console.error('Prediction error:', e);
      setError('Failed to make prediction. Please check your inputs.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setFormData({
      pregnancies: '',
      glucose: '',
      blood_pressure: '',
      skin_thickness: '',
      insulin: '',
      bmi: '',
      diabetes_pedigree_function: '',
      age: ''
    });
    setPrediction(null);
    setError(null);
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8 px-4">
      <div className="max-w-6xl mx-auto">
        <div className="text-center mb-8">
          <h1 className="text-3xl font-bold text-gray-900 mb-2">Diabetes Prediction System</h1>
          <p className="text-gray-600">Machine Learning Model Comparison</p>
        </div>

        {metrics && (
          <Card className="mb-8" data-testid="model-metrics-card">
            <CardHeader>
              <CardTitle>Model Performance Metrics</CardTitle>
              <CardDescription>Comparison of Logistic Regression and Random Forest models on Pima Indians Diabetes Dataset</CardDescription>
            </CardHeader>
            <CardContent>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
                <div data-testid="logistic-regression-metrics">
                  <h3 className="font-semibold text-lg mb-3">Logistic Regression</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Accuracy:</span>
                      <span className="font-medium">{metrics.logistic_regression.accuracy}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Precision:</span>
                      <span className="font-medium">{metrics.logistic_regression.precision}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Recall:</span>
                      <span className="font-medium">{metrics.logistic_regression.recall}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">F1-Score:</span>
                      <span className="font-medium">{metrics.logistic_regression.f1_score}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">ROC-AUC:</span>
                      <span className="font-medium">{metrics.logistic_regression.roc_auc}</span>
                    </div>
                  </div>
                </div>
                <div data-testid="random-forest-metrics">
                  <h3 className="font-semibold text-lg mb-3">Random Forest</h3>
                  <div className="space-y-2">
                    <div className="flex justify-between">
                      <span className="text-gray-600">Accuracy:</span>
                      <span className="font-medium">{metrics.random_forest.accuracy}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Precision:</span>
                      <span className="font-medium">{metrics.random_forest.precision}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">Recall:</span>
                      <span className="font-medium">{metrics.random_forest.recall}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">F1-Score:</span>
                      <span className="font-medium">{metrics.random_forest.f1_score}</span>
                    </div>
                    <div className="flex justify-between">
                      <span className="text-gray-600">ROC-AUC:</span>
                      <span className="font-medium">{metrics.random_forest.roc_auc}</span>
                    </div>
                  </div>
                </div>
              </div>
            </CardContent>
          </Card>
        )}

        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          <Card data-testid="prediction-form-card">
            <CardHeader>
              <CardTitle>Enter Patient Data</CardTitle>
              <CardDescription>Fill in the medical parameters for prediction</CardDescription>
            </CardHeader>
            <CardContent>
              <form onSubmit={handleSubmit} className="space-y-4">
                <div>
                  <Label htmlFor="pregnancies">Pregnancies</Label>
                  <Input
                    id="pregnancies"
                    name="pregnancies"
                    type="number"
                    step="1"
                    value={formData.pregnancies}
                    onChange={handleInputChange}
                    required
                    data-testid="input-pregnancies"
                  />
                </div>
                <div>
                  <Label htmlFor="glucose">Glucose (mg/dL)</Label>
                  <Input
                    id="glucose"
                    name="glucose"
                    type="number"
                    step="0.1"
                    value={formData.glucose}
                    onChange={handleInputChange}
                    required
                    data-testid="input-glucose"
                  />
                </div>
                <div>
                  <Label htmlFor="blood_pressure">Blood Pressure (mm Hg)</Label>
                  <Input
                    id="blood_pressure"
                    name="blood_pressure"
                    type="number"
                    step="0.1"
                    value={formData.blood_pressure}
                    onChange={handleInputChange}
                    required
                    data-testid="input-blood-pressure"
                  />
                </div>
                <div>
                  <Label htmlFor="skin_thickness">Skin Thickness (mm)</Label>
                  <Input
                    id="skin_thickness"
                    name="skin_thickness"
                    type="number"
                    step="0.1"
                    value={formData.skin_thickness}
                    onChange={handleInputChange}
                    required
                    data-testid="input-skin-thickness"
                  />
                </div>
                <div>
                  <Label htmlFor="insulin">Insulin (μU/mL)</Label>
                  <Input
                    id="insulin"
                    name="insulin"
                    type="number"
                    step="0.1"
                    value={formData.insulin}
                    onChange={handleInputChange}
                    required
                    data-testid="input-insulin"
                  />
                </div>
                <div>
                  <Label htmlFor="bmi">BMI</Label>
                  <Input
                    id="bmi"
                    name="bmi"
                    type="number"
                    step="0.1"
                    value={formData.bmi}
                    onChange={handleInputChange}
                    required
                    data-testid="input-bmi"
                  />
                </div>
                <div>
                  <Label htmlFor="diabetes_pedigree_function">Diabetes Pedigree Function</Label>
                  <Input
                    id="diabetes_pedigree_function"
                    name="diabetes_pedigree_function"
                    type="number"
                    step="0.001"
                    value={formData.diabetes_pedigree_function}
                    onChange={handleInputChange}
                    required
                    data-testid="input-diabetes-pedigree"
                  />
                </div>
                <div>
                  <Label htmlFor="age">Age (years)</Label>
                  <Input
                    id="age"
                    name="age"
                    type="number"
                    step="1"
                    value={formData.age}
                    onChange={handleInputChange}
                    required
                    data-testid="input-age"
                  />
                </div>
                <div className="flex gap-2">
                  <Button type="submit" disabled={loading} className="flex-1" data-testid="predict-button">
                    {loading ? 'Predicting...' : 'Predict'}
                  </Button>
                  <Button type="button" variant="outline" onClick={handleReset} data-testid="reset-button">
                    Reset
                  </Button>
                </div>
              </form>
            </CardContent>
          </Card>

          <div className="space-y-6">
            {error && (
              <Alert variant="destructive" data-testid="error-alert">
                <AlertDescription>{error}</AlertDescription>
              </Alert>
            )}

            {prediction && (
              <Card data-testid="prediction-results-card">
                <CardHeader>
                  <CardTitle>Prediction Results</CardTitle>
                </CardHeader>
                <CardContent className="space-y-6">
                  <div data-testid="logistic-regression-result">
                    <h3 className="font-semibold text-lg mb-3">Logistic Regression</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Result:</span>
                        <span className={`font-bold ${prediction.logistic_regression.prediction === 1 ? 'text-red-600' : 'text-green-600'}`}>
                          {prediction.logistic_regression.result}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Probability (No Diabetes):</span>
                        <span className="font-medium">{(prediction.logistic_regression.probability_no_diabetes * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Probability (Diabetes):</span>
                        <span className="font-medium">{(prediction.logistic_regression.probability_diabetes * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                  </div>
                  
                  <Separator />
                  
                  <div data-testid="random-forest-result">
                    <h3 className="font-semibold text-lg mb-3">Random Forest</h3>
                    <div className="space-y-2">
                      <div className="flex justify-between">
                        <span className="text-gray-600">Result:</span>
                        <span className={`font-bold ${prediction.random_forest.prediction === 1 ? 'text-red-600' : 'text-green-600'}`}>
                          {prediction.random_forest.result}
                        </span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Probability (No Diabetes):</span>
                        <span className="font-medium">{(prediction.random_forest.probability_no_diabetes * 100).toFixed(2)}%</span>
                      </div>
                      <div className="flex justify-between">
                        <span className="text-gray-600">Probability (Diabetes):</span>
                        <span className="font-medium">{(prediction.random_forest.probability_diabetes * 100).toFixed(2)}%</span>
                      </div>
                    </div>
                  </div>
                </CardContent>
              </Card>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;