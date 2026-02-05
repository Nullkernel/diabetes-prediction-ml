import requests
import sys
import json
from datetime import datetime

class DiabetesPredictionTester:
    def __init__(self, base_url="https://ml-diabetes-detect.preview.emergentagent.com"):
        self.base_url = base_url
        self.api_url = f"{base_url}/api"
        self.tests_run = 0
        self.tests_passed = 0
        self.test_results = []

    def log_test(self, name, success, details=""):
        """Log test result"""
        self.tests_run += 1
        if success:
            self.tests_passed += 1
        
        result = {
            "test_name": name,
            "success": success,
            "details": details,
            "timestamp": datetime.now().isoformat()
        }
        self.test_results.append(result)
        
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status} - {name}")
        if details:
            print(f"   Details: {details}")

    def test_api_root(self):
        """Test API root endpoint"""
        try:
            response = requests.get(f"{self.api_url}/", timeout=10)
            success = response.status_code == 200
            details = f"Status: {response.status_code}"
            if success:
                data = response.json()
                details += f", Response: {data}"
            self.log_test("API Root Endpoint", success, details)
            return success
        except Exception as e:
            self.log_test("API Root Endpoint", False, f"Error: {str(e)}")
            return False

    def test_model_metrics(self):
        """Test GET /api/models/metrics endpoint"""
        try:
            response = requests.get(f"{self.api_url}/models/metrics", timeout=15)
            success = response.status_code == 200
            
            if success:
                data = response.json()
                # Verify structure
                required_models = ['logistic_regression', 'random_forest']
                required_metrics = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
                
                structure_valid = True
                for model in required_models:
                    if model not in data:
                        structure_valid = False
                        break
                    for metric in required_metrics:
                        if metric not in data[model]:
                            structure_valid = False
                            break
                
                if structure_valid:
                    details = f"Status: {response.status_code}, Models: {list(data.keys())}"
                    details += f", LR Accuracy: {data['logistic_regression']['accuracy']}"
                    details += f", RF Accuracy: {data['random_forest']['accuracy']}"
                else:
                    success = False
                    details = f"Invalid structure: {data}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text}"
            
            self.log_test("Model Metrics Endpoint", success, details)
            return success, data if success else None
            
        except Exception as e:
            self.log_test("Model Metrics Endpoint", False, f"Error: {str(e)}")
            return False, None

    def test_prediction_valid_input(self):
        """Test POST /api/predict with valid input"""
        try:
            # Sample valid input based on Pima Indians dataset
            test_data = {
                "pregnancies": 6,
                "glucose": 148,
                "blood_pressure": 72,
                "skin_thickness": 35,
                "insulin": 0,
                "bmi": 33.6,
                "diabetes_pedigree_function": 0.627,
                "age": 50
            }
            
            response = requests.post(
                f"{self.api_url}/predict", 
                json=test_data, 
                headers={'Content-Type': 'application/json'},
                timeout=15
            )
            
            success = response.status_code == 200
            
            if success:
                data = response.json()
                # Verify prediction structure
                required_models = ['logistic_regression', 'random_forest']
                required_fields = ['prediction', 'probability_no_diabetes', 'probability_diabetes', 'result']
                
                structure_valid = True
                for model in required_models:
                    if model not in data:
                        structure_valid = False
                        break
                    for field in required_fields:
                        if field not in data[model]:
                            structure_valid = False
                            break
                
                if structure_valid:
                    lr_result = data['logistic_regression']['result']
                    rf_result = data['random_forest']['result']
                    details = f"Status: {response.status_code}, LR: {lr_result}, RF: {rf_result}"
                else:
                    success = False
                    details = f"Invalid prediction structure: {data}"
            else:
                details = f"Status: {response.status_code}, Response: {response.text}"
            
            self.log_test("Prediction Valid Input", success, details)
            return success, data if success else None
            
        except Exception as e:
            self.log_test("Prediction Valid Input", False, f"Error: {str(e)}")
            return False, None

    def test_prediction_edge_cases(self):
        """Test prediction with edge cases"""
        test_cases = [
            {
                "name": "All zeros",
                "data": {
                    "pregnancies": 0, "glucose": 0, "blood_pressure": 0,
                    "skin_thickness": 0, "insulin": 0, "bmi": 0,
                    "diabetes_pedigree_function": 0, "age": 0
                }
            },
            {
                "name": "High values",
                "data": {
                    "pregnancies": 15, "glucose": 200, "blood_pressure": 120,
                    "skin_thickness": 50, "insulin": 300, "bmi": 45,
                    "diabetes_pedigree_function": 2.0, "age": 80
                }
            }
        ]
        
        all_passed = True
        for test_case in test_cases:
            try:
                response = requests.post(
                    f"{self.api_url}/predict", 
                    json=test_case["data"], 
                    headers={'Content-Type': 'application/json'},
                    timeout=15
                )
                
                success = response.status_code == 200
                if success:
                    data = response.json()
                    details = f"Status: {response.status_code}, Case: {test_case['name']}"
                else:
                    details = f"Status: {response.status_code}, Case: {test_case['name']}, Error: {response.text}"
                    all_passed = False
                
                self.log_test(f"Prediction Edge Case - {test_case['name']}", success, details)
                
            except Exception as e:
                self.log_test(f"Prediction Edge Case - {test_case['name']}", False, f"Error: {str(e)}")
                all_passed = False
        
        return all_passed

    def test_prediction_invalid_input(self):
        """Test prediction with invalid input"""
        try:
            # Missing required field
            invalid_data = {
                "pregnancies": 6,
                "glucose": 148,
                # Missing other required fields
            }
            
            response = requests.post(
                f"{self.api_url}/predict", 
                json=invalid_data, 
                headers={'Content-Type': 'application/json'},
                timeout=15
            )
            
            # Should return 422 for validation error
            success = response.status_code == 422
            details = f"Status: {response.status_code} (Expected 422 for validation error)"
            
            self.log_test("Prediction Invalid Input", success, details)
            return success
            
        except Exception as e:
            self.log_test("Prediction Invalid Input", False, f"Error: {str(e)}")
            return False

    def run_all_tests(self):
        """Run all backend tests"""
        print("🔍 Starting Diabetes Prediction API Tests...")
        print(f"Testing against: {self.base_url}")
        print("=" * 60)
        
        # Test API availability
        if not self.test_api_root():
            print("❌ API not accessible, stopping tests")
            return False
        
        # Test model metrics
        metrics_success, metrics_data = self.test_model_metrics()
        if not metrics_success:
            print("❌ Model metrics failed, stopping tests")
            return False
        
        # Test predictions
        prediction_success, prediction_data = self.test_prediction_valid_input()
        if not prediction_success:
            print("❌ Basic prediction failed, stopping tests")
            return False
        
        # Test edge cases
        self.test_prediction_edge_cases()
        
        # Test invalid input
        self.test_prediction_invalid_input()
        
        # Print summary
        print("\n" + "=" * 60)
        print(f"📊 Test Summary: {self.tests_passed}/{self.tests_run} tests passed")
        print(f"Success Rate: {(self.tests_passed/self.tests_run)*100:.1f}%")
        
        if self.tests_passed == self.tests_run:
            print("🎉 All tests passed!")
            return True
        else:
            print("⚠️  Some tests failed")
            return False

def main():
    tester = DiabetesPredictionTester()
    success = tester.run_all_tests()
    
    # Save detailed results
    with open('/app/backend_test_results.json', 'w') as f:
        json.dump({
            'summary': {
                'total_tests': tester.tests_run,
                'passed_tests': tester.tests_passed,
                'success_rate': (tester.tests_passed/tester.tests_run)*100 if tester.tests_run > 0 else 0,
                'timestamp': datetime.now().isoformat()
            },
            'detailed_results': tester.test_results
        }, f, indent=2)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main())