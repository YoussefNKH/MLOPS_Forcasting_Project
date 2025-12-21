from pydantic import BaseModel

class PredictionOutput(BaseModel):
    prediction: float
    model_name: str
    model_version: str