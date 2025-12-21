import pytest
from unittest.mock import patch, MagicMock
import sys
import os

# Add the frontend directory to the path so we can import from streamlit_app
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'frontend'))

# Import the helper functions from streamlit_app
# Note: We need to mock streamlit before importing
sys.modules['streamlit'] = MagicMock()

from streamlit_app import check_api_health, get_model_info, make_prediction, make_batch_prediction


class TestCheckApiHealth:
    """Test cases for check_api_health function"""
    
    @patch('streamlit_app.requests.get')
    def test_api_health_success(self, mock_get):
        """Test when API is healthy"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_get.return_value = mock_response
        
        result = check_api_health()
        
        assert result is True
        mock_get.assert_called_once_with("http://backend-api:8000/api/health", timeout=5)
    
    @patch('streamlit_app.requests.get')
    def test_api_health_failure_status(self, mock_get):
        """Test when API returns non-200 status"""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_get.return_value = mock_response
        
        result = check_api_health()
        
        assert result is False
    
    @patch('streamlit_app.requests.get')
    def test_api_health_exception(self, mock_get):
        """Test when API request raises an exception"""
        # Mock exception
        mock_get.side_effect = Exception("Connection error")
        
        result = check_api_health()
        
        assert result is False


class TestGetModelInfo:
    """Test cases for get_model_info function"""
    
    @patch('streamlit_app.requests.get')
    def test_get_model_info_success(self, mock_get):
        """Test successful model info retrieval"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'model_info': {
                'name': 'BestRegressionModel',
                'version': '1',
                'stage': 'Production'
            }
        }
        mock_get.return_value = mock_response
        
        result = get_model_info()
        
        assert result is not None
        assert 'model_info' in result
        assert result['model_info']['name'] == 'BestRegressionModel'
        mock_get.assert_called_once_with("http://backend-api:8000/api/model-info", timeout=5)
    
    @patch('streamlit_app.requests.get')
    def test_get_model_info_failure(self, mock_get):
        """Test when model info request fails"""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 404
        mock_get.return_value = mock_response
        
        result = get_model_info()
        
        assert result is None
    
    @patch('streamlit_app.requests.get')
    def test_get_model_info_exception(self, mock_get):
        """Test when model info request raises an exception"""
        # Mock exception
        mock_get.side_effect = Exception("Connection error")
        
        result = get_model_info()
        
        assert result is None


class TestMakePrediction:
    """Test cases for make_prediction function"""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.requests.post')
    def test_make_prediction_success(self, mock_post, mock_st):
        """Test successful prediction"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'prediction': 42.5,
            'model_name': 'BestRegressionModel',
            'model_version': '1'
        }
        mock_post.return_value = mock_response
        
        input_data = {'feature1': 1, 'feature2': 2}
        result = make_prediction(input_data)
        
        assert result is not None
        assert result['prediction'] == 42.5
        assert result['model_name'] == 'BestRegressionModel'
        mock_post.assert_called_once_with(
            "http://backend-api:8000/api/predict",
            json=input_data,
            timeout=10
        )
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.requests.post')
    def test_make_prediction_failure(self, mock_post, mock_st):
        """Test when prediction request fails"""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {'detail': 'Prediction failed'}
        mock_post.return_value = mock_response
        
        input_data = {'feature1': 1, 'feature2': 2}
        result = make_prediction(input_data)
        
        assert result is None
        # Verify that st.error was called
        mock_st.error.assert_called_once()
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.requests.post')
    def test_make_prediction_exception(self, mock_post, mock_st):
        """Test when prediction request raises an exception"""
        # Mock exception
        mock_post.side_effect = Exception("Connection error")
        
        input_data = {'feature1': 1, 'feature2': 2}
        result = make_prediction(input_data)
        
        assert result is None
        # Verify that st.error was called
        mock_st.error.assert_called_once()


class TestMakeBatchPrediction:
    """Test cases for make_batch_prediction function"""
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.requests.post')
    def test_batch_prediction_success(self, mock_post, mock_st):
        """Test successful batch prediction"""
        # Mock successful response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            'predictions': [42.5, 38.2, 45.1],
            'model_name': 'BestRegressionModel',
            'model_version': '1'
        }
        mock_post.return_value = mock_response
        
        data_list = [
            {'feature1': 1, 'feature2': 2},
            {'feature1': 3, 'feature2': 4},
            {'feature1': 5, 'feature2': 6}
        ]
        result = make_batch_prediction(data_list)
        
        assert result is not None
        assert len(result['predictions']) == 3
        assert result['predictions'] == [42.5, 38.2, 45.1]
        mock_post.assert_called_once_with(
            "http://backend-api:8000/api/predict-batch",
            json={"data": data_list},
            timeout=30
        )
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.requests.post')
    def test_batch_prediction_failure(self, mock_post, mock_st):
        """Test when batch prediction request fails"""
        # Mock failed response
        mock_response = MagicMock()
        mock_response.status_code = 500
        mock_response.json.return_value = {'detail': 'Batch prediction failed'}
        mock_post.return_value = mock_response
        
        data_list = [{'feature1': 1, 'feature2': 2}]
        result = make_batch_prediction(data_list)
        
        assert result is None
        # Verify that st.error was called
        mock_st.error.assert_called_once()
    
    @patch('streamlit_app.st')
    @patch('streamlit_app.requests.post')
    def test_batch_prediction_exception(self, mock_post, mock_st):
        """Test when batch prediction request raises an exception"""
        # Mock exception
        mock_post.side_effect = Exception("Connection error")
        
        data_list = [{'feature1': 1, 'feature2': 2}]
        result = make_batch_prediction(data_list)
        
        assert result is None
        # Verify that st.error was called
        mock_st.error.assert_called_once()
