import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock
import sys

# Mock the ML libraries before importing trainer
sys.modules['lightgbm'] = MagicMock()
sys.modules['catboost'] = MagicMock()
sys.modules['xgboost'] = MagicMock()

from src.train.trainer import train


class TestTrainer:
    """Test cases for train function"""
    
    @patch('src.train.trainer.train_lgbm')
    def test_train_lgbm(self, mock_train_lgbm):
        """Test that train dispatches to train_lgbm for lgbm model"""
        # Create mock data
        X_train = pd.DataFrame({'feature1': [1, 2, 3]})
        y_train = pd.Series([10, 20, 30])
        X_valid = pd.DataFrame({'feature1': [4, 5]})
        y_valid = pd.Series([40, 50])
        common_params = {'n_estimators': 100, 'learning_rate': 0.1}
        
        # Create mock model
        mock_model = MagicMock()
        mock_train_lgbm.return_value = mock_model
        
        # Call train
        result = train('lgbm', X_train, y_train, X_valid, y_valid, common_params)
        
        # Verify train_lgbm was called with correct parameters
        mock_train_lgbm.assert_called_once_with(
            X_train, y_train, X_valid, y_valid, common_params
        )
        
        # Verify the result is the mock model
        assert result == mock_model
    
    @patch('src.train.trainer.train_catboost')
    def test_train_catboost(self, mock_train_catboost):
        """Test that train dispatches to train_catboost for catboost model"""
        # Create mock data
        X_train = pd.DataFrame({'feature1': [1, 2, 3]})
        y_train = pd.Series([10, 20, 30])
        X_valid = pd.DataFrame({'feature1': [4, 5]})
        y_valid = pd.Series([40, 50])
        
        # Create mock model
        mock_model = MagicMock()
        mock_train_catboost.return_value = mock_model
        
        # Call train
        result = train('catboost', X_train, y_train, X_valid, y_valid)
        
        # Verify train_catboost was called with correct parameters
        mock_train_catboost.assert_called_once_with(
            X_train, y_train, X_valid, y_valid
        )
        
        # Verify the result is the mock model
        assert result == mock_model
    
    @patch('src.train.trainer.train_xgboost')
    def test_train_xgboost(self, mock_train_xgboost):
        """Test that train dispatches to train_xgboost for xgboost model"""
        # Create mock data
        X_train = pd.DataFrame({'feature1': [1, 2, 3]})
        y_train = pd.Series([10, 20, 30])
        X_valid = pd.DataFrame({'feature1': [4, 5]})
        y_valid = pd.Series([40, 50])
        common_params = {'n_estimators': 100, 'learning_rate': 0.1}
        
        # Create mock model
        mock_model = MagicMock()
        mock_train_xgboost.return_value = mock_model
        
        # Call train
        result = train('xgboost', X_train, y_train, X_valid, y_valid, common_params)
        
        # Verify train_xgboost was called with correct parameters
        mock_train_xgboost.assert_called_once_with(
            X_train, y_train, X_valid, y_valid, common_params
        )
        
        # Verify the result is the mock model
        assert result == mock_model
    
    def test_train_unknown_model(self):
        """Test that train raises ValueError for unknown model"""
        # Create mock data
        X_train = pd.DataFrame({'feature1': [1, 2, 3]})
        y_train = pd.Series([10, 20, 30])
        X_valid = pd.DataFrame({'feature1': [4, 5]})
        y_valid = pd.Series([40, 50])
        
        # Call train with unknown model name
        with pytest.raises(ValueError, match="Unknown model"):
            train('unknown_model', X_train, y_train, X_valid, y_valid)
