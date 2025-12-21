from fastapi import APIRouter
import mlflow
import mlflow.pyfunc
import pandas as pd 
from app.utils.load_model import load_best_model_from_mlflow
from fastapi import HTTPException
from app.models.batch_prediction_input import BatchPredictionInput
from app.models.batch_prediction_output import BatchPredictionOutput
from app.models.config import Config
from app.models.prediction_input import PredictionInput
from app.models.prediction_output import PredictionOutput


api_router = APIRouter(
    tags=["Endpoints"]
)

loaded_model = None
model_info= {}

@api_router.get("/model-info", summary="Model Info Endpoint")
async def get_model_info():
    if loaded_model is None:
        raise HTTPException(status_code=404, detail="Model not loaded")
    return {
        "status":"loaded",
        "info": model_info
    }
    
@api_router.post("/predict", response_model=PredictionOutput, summary="Make Sales Prediction")
async def predict(input_data: PredictionInput):
    """
    Make a sales prediction using the loaded model.
    
    Args:
        input_data: Input features for prediction
    
    Returns:
        Prediction result
    """
    global loaded_model, model_info
    
    # Load model if not already loaded
    if loaded_model is None:
        try:
            loaded_model, model_info = load_best_model_from_mlflow()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded. Please call /load-model endpoint first. Error: {str(e)}"
            )
    
    try:
        # Convert input to DataFrame
        input_dict = input_data.model_dump()
        input_df = pd.DataFrame([input_dict])
        
        # Make prediction
        prediction = loaded_model.predict(input_df)
        
        # Return result
        return PredictionOutput(
            prediction=float(prediction[0]),
            model_name=model_info.get("name", "unknown"),
            model_version=str(model_info.get("version", "unknown"))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@api_router.post("/predict-batch", response_model=BatchPredictionOutput, summary="Make Batch Sales Predictions")
async def predict_batch(input_data: BatchPredictionInput):

    global loaded_model, model_info
    
    # Load model if not already loaded
    if loaded_model is None:
        try:
            loaded_model, model_info = load_best_model_from_mlflow()
        except Exception as e:
            raise HTTPException(
                status_code=503,
                detail=f"Model not loaded. Please call /load-model endpoint first. Error: {str(e)}"
            )
    
    try:
        # Convert input to DataFrame
        input_list = [item.model_dump() for item in input_data.data]
        input_df = pd.DataFrame(input_list)
        
        # Make predictions
        predictions = loaded_model.predict(input_df)
        
        # Return results
        return BatchPredictionOutput(
            predictions=[float(pred) for pred in predictions],
            model_name=model_info.get("name", "unknown"),
            model_version=str(model_info.get("version", "unknown"))
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch prediction failed: {str(e)}")

@api_router.get("/health", summary="Health Check Endpoint")
async def health_check():
    return {"status": "ok"}
