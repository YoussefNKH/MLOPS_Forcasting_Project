from pydantic import BaseModel
from typing import List

class BatchPredictionOutput(BaseModel):
    predictions: List[float]
    model_name: str
    model_version: str
    