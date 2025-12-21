import pytest
import pandas as pd
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock
import sys

# Mock mlflow before importing app
sys.modules['mlflow'] = MagicMock()
sys.modules['mlflow.sklearn'] = MagicMock()
sys.modules['mlflow.pyfunc'] = MagicMock()

from app.main import app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app"""
    return TestClient(app)


@pytest.fixture
def mock_model_and_info():
    """Create mock model and info for testing"""
    mock_model = MagicMock()
    mock_model.predict.return_value = [42.5]
    
    mock_info = {
        'name': 'BestRegressionModel',
        'version': '1',
        'stage': 'Production',
        'run_id': 'test_run_id_12345'
    }
    
    return mock_model, mock_info


class TestHealthEndpoint:
    """Test cases for /health endpoint"""
    
    def test_health_check(self, client):
        """Test that health endpoint returns OK"""
        response = client.get("/api/health")
        
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}


class TestModelInfoEndpoint:
    """Test cases for /model-info endpoint"""
    
    @patch('app.api.endpoints.loaded_model')
    @patch('app.api.endpoints.model_info')
    def test_model_info_success(self, mock_info_var, mock_model_var, client, mock_model_and_info):
        """Test model info endpoint when model is loaded"""
        mock_model, mock_info = mock_model_and_info
        
        # Set the module-level variables
        mock_model_var = mock_model
        mock_info_var = mock_info
        
        # Patch the endpoints module to return our mock
        with patch('app.api.endpoints.loaded_model', mock_model):
            with patch('app.api.endpoints.model_info', mock_info):
                response = client.get("/api/model-info")
        
        assert response.status_code == 200
        data = response.json()
        assert data['status'] == 'loaded'
        assert 'info' in data
    
    @patch('app.api.endpoints.loaded_model', None)
    def test_model_info_not_loaded(self, client):
        """Test model info endpoint when model is not loaded"""
        response = client.get("/api/model-info")
        
        assert response.status_code == 404
        assert 'Model not loaded' in response.json()['detail']


class TestPredictEndpoint:
    """Test cases for /predict endpoint"""
    
    @patch('app.api.endpoints.loaded_model')
    @patch('app.api.endpoints.model_info')
    def test_predict_success(self, mock_info_var, mock_model_var, client, mock_model_and_info):
        """Test successful prediction"""
        mock_model, mock_info = mock_model_and_info
        
        # Sample input data
        input_data = {
            "id": 1,
            "item_id": 1001,
            "dept_id": 1,
            "cat_id": 1,
            "store_id": 1,
            "state_id": 1,
            "d": 1000,
            "wm_yr_wk": 11500,
            "weekday": 1,
            "wday": 2,
            "month": 6,
            "year": 2016,
            "event_name_1": 0,
            "event_type_1": 0,
            "event_name_2": 0,
            "event_type_2": 0,
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
            "sell_price": 3.97,
            "revenue": 11.91,
            "sold_lag_1": 3.0,
            "sold_lag_2": 2.0,
            "sold_lag_3": 1.0,
            "sold_lag_6": 4.0,
            "sold_lag_12": 2.5,
            "sold_lag_24": 3.0,
            "sold_lag_36": 2.8,
            "iteam_sold_avg": 2.5,
            "state_sold_avg": 150.0,
            "store_sold_avg": 50.0,
            "cat_sold_avg": 75.0,
            "dept_sold_avg": 30.0,
            "cat_dept_sold_avg": 25.0,
            "store_item_sold_avg": 2.3,
            "cat_item_sold_avg": 2.4,
            "dept_item_sold_avg": 2.6,
            "state_store_sold_avg": 45.0,
            "state_store_cat_sold_avg": 22.0,
            "store_cat_dept_sold_avg": 18.0,
            "rolling_sold_mean": 2.7,
            "expanding_sold_mean": 2.5,
            "selling_trend": 0.05
        }
        
        with patch('app.api.endpoints.loaded_model', mock_model):
            with patch('app.api.endpoints.model_info', mock_info):
                response = client.post("/api/predict", json=input_data)
        
        assert response.status_code == 200
        data = response.json()
        assert 'prediction' in data
        assert 'model_name' in data
        assert 'model_version' in data
        assert data['prediction'] == 42.5
    
    @patch('app.api.endpoints.loaded_model', None)
    @patch('app.api.endpoints.load_best_model_from_mlflow')
    def test_predict_model_not_loaded(self, mock_load_model, client):
        """Test prediction when model is not initially loaded"""
        mock_load_model.side_effect = Exception("Failed to load model")
        
        input_data = {
            "id": 1,
            "item_id": 1001,
            "dept_id": 1,
            "cat_id": 1,
            "store_id": 1,
            "state_id": 1,
            "d": 1000,
            "wm_yr_wk": 11500,
            "weekday": 1,
            "wday": 2,
            "month": 6,
            "year": 2016,
            "event_name_1": 0,
            "event_type_1": 0,
            "event_name_2": 0,
            "event_type_2": 0,
            "snap_CA": 0,
            "snap_TX": 0,
            "snap_WI": 0,
            "sell_price": 3.97,
            "revenue": 11.91,
            "sold_lag_1": 3.0,
            "sold_lag_2": 2.0,
            "sold_lag_3": 1.0,
            "sold_lag_6": 4.0,
            "sold_lag_12": 2.5,
            "sold_lag_24": 3.0,
            "sold_lag_36": 2.8,
            "iteam_sold_avg": 2.5,
            "state_sold_avg": 150.0,
            "store_sold_avg": 50.0,
            "cat_sold_avg": 75.0,
            "dept_sold_avg": 30.0,
            "cat_dept_sold_avg": 25.0,
            "store_item_sold_avg": 2.3,
            "cat_item_sold_avg": 2.4,
            "dept_item_sold_avg": 2.6,
            "state_store_sold_avg": 45.0,
            "state_store_cat_sold_avg": 22.0,
            "store_cat_dept_sold_avg": 18.0,
            "rolling_sold_mean": 2.7,
            "expanding_sold_mean": 2.5,
            "selling_trend": 0.05
        }
        
        response = client.post("/api/predict", json=input_data)
        
        assert response.status_code == 503


class TestBatchPredictEndpoint:
    """Test cases for /predict-batch endpoint"""
    
    @patch('app.api.endpoints.loaded_model')
    @patch('app.api.endpoints.model_info')
    def test_batch_predict_success(self, mock_info_var, mock_model_var, client, mock_model_and_info):
        """Test successful batch prediction"""
        mock_model, mock_info = mock_model_and_info
        mock_model.predict.return_value = [42.5, 38.2]
        
        # Sample batch input data
        input_data = {
            "data": [
                {
                    "id": 1, "item_id": 1001, "dept_id": 1, "cat_id": 1,
                    "store_id": 1, "state_id": 1, "d": 1000, "wm_yr_wk": 11500,
                    "weekday": 1, "wday": 2, "month": 6, "year": 2016,
                    "event_name_1": 0, "event_type_1": 0, "event_name_2": 0,
                    "event_type_2": 0, "snap_CA": 0, "snap_TX": 0, "snap_WI": 0,
                    "sell_price": 3.97, "revenue": 11.91, "sold_lag_1": 3.0,
                    "sold_lag_2": 2.0, "sold_lag_3": 1.0, "sold_lag_6": 4.0,
                    "sold_lag_12": 2.5, "sold_lag_24": 3.0, "sold_lag_36": 2.8,
                    "iteam_sold_avg": 2.5, "state_sold_avg": 150.0,
                    "store_sold_avg": 50.0, "cat_sold_avg": 75.0,
                    "dept_sold_avg": 30.0, "cat_dept_sold_avg": 25.0,
                    "store_item_sold_avg": 2.3, "cat_item_sold_avg": 2.4,
                    "dept_item_sold_avg": 2.6, "state_store_sold_avg": 45.0,
                    "state_store_cat_sold_avg": 22.0,
                    "store_cat_dept_sold_avg": 18.0, "rolling_sold_mean": 2.7,
                    "expanding_sold_mean": 2.5, "selling_trend": 0.05
                },
                {
                    "id": 2, "item_id": 1002, "dept_id": 1, "cat_id": 1,
                    "store_id": 1, "state_id": 1, "d": 1001, "wm_yr_wk": 11500,
                    "weekday": 2, "wday": 3, "month": 6, "year": 2016,
                    "event_name_1": 0, "event_type_1": 0, "event_name_2": 0,
                    "event_type_2": 0, "snap_CA": 0, "snap_TX": 0, "snap_WI": 0,
                    "sell_price": 4.50, "revenue": 13.50, "sold_lag_1": 2.5,
                    "sold_lag_2": 2.2, "sold_lag_3": 1.5, "sold_lag_6": 3.8,
                    "sold_lag_12": 2.3, "sold_lag_24": 2.9, "sold_lag_36": 2.6,
                    "iteam_sold_avg": 2.4, "state_sold_avg": 148.0,
                    "store_sold_avg": 49.0, "cat_sold_avg": 74.0,
                    "dept_sold_avg": 29.0, "cat_dept_sold_avg": 24.5,
                    "store_item_sold_avg": 2.2, "cat_item_sold_avg": 2.3,
                    "dept_item_sold_avg": 2.5, "state_store_sold_avg": 44.0,
                    "state_store_cat_sold_avg": 21.5,
                    "store_cat_dept_sold_avg": 17.5, "rolling_sold_mean": 2.6,
                    "expanding_sold_mean": 2.4, "selling_trend": 0.04
                }
            ]
        }
        
        with patch('app.api.endpoints.loaded_model', mock_model):
            with patch('app.api.endpoints.model_info', mock_info):
                response = client.post("/api/predict-batch", json=input_data)
        
        assert response.status_code == 200
        data = response.json()
        assert 'predictions' in data
        assert len(data['predictions']) == 2
        assert data['predictions'] == [42.5, 38.2]
