import pytest
import pandas as pd
import numpy as np
from unittest.mock import MagicMock
from src.evaluate import evaluate_model


class TestEvaluateModel:
    """Test cases for evaluate_model function"""
    
    def test_evaluate_model_perfect_prediction(self):
        """Test evaluation with perfect predictions"""
        # Create mock model that predicts perfectly
        mock_model = MagicMock()
        X_valid = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y_valid = pd.Series([10, 20, 30, 40, 50])
        
        # Mock perfect predictions
        mock_model.predict.return_value = np.array([10, 20, 30, 40, 50])
        
        metrics = evaluate_model(mock_model, X_valid, y_valid)
        
        # Perfect predictions should have 0 error and R2 = 1
        assert metrics['rmse'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['mse'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['mae'] == pytest.approx(0.0, abs=1e-10)
        assert metrics['r2'] == pytest.approx(1.0, abs=1e-10)
    
    def test_evaluate_model_with_errors(self):
        """Test evaluation with prediction errors"""
        # Create mock model
        mock_model = MagicMock()
        X_valid = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y_valid = pd.Series([10, 20, 30, 40, 50])
        
        # Mock predictions with some error
        mock_model.predict.return_value = np.array([12, 18, 32, 38, 52])
        
        metrics = evaluate_model(mock_model, X_valid, y_valid)
        
        # Check that metrics are calculated
        assert 'rmse' in metrics
        assert 'mse' in metrics
        assert 'mae' in metrics
        assert 'r2' in metrics
        
        # Check that errors are positive
        assert metrics['rmse'] > 0
        assert metrics['mse'] > 0
        assert metrics['mae'] > 0
        
        # R2 should be less than 1 but still positive for reasonable predictions
        assert 0 < metrics['r2'] < 1
    
    def test_evaluate_model_return_dict(self):
        """Test that evaluate_model returns a dictionary with correct keys"""
        mock_model = MagicMock()
        X_valid = pd.DataFrame({'feature1': [1, 2, 3]})
        y_valid = pd.Series([10, 20, 30])
        
        mock_model.predict.return_value = np.array([11, 19, 31])
        
        metrics = evaluate_model(mock_model, X_valid, y_valid)
        
        # Check that result is a dictionary
        assert isinstance(metrics, dict)
        
        # Check that all expected keys are present
        expected_keys = {'rmse', 'mse', 'mae', 'r2'}
        assert set(metrics.keys()) == expected_keys
    
    def test_evaluate_model_rmse_mse_relationship(self):
        """Test that RMSE is the square root of MSE"""
        mock_model = MagicMock()
        X_valid = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y_valid = pd.Series([10, 20, 30, 40, 50])
        
        mock_model.predict.return_value = np.array([12, 22, 28, 42, 48])
        
        metrics = evaluate_model(mock_model, X_valid, y_valid)
        
        # RMSE should be sqrt of MSE
        assert metrics['rmse'] == pytest.approx(np.sqrt(metrics['mse']), abs=1e-10)
