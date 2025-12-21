from pydantic import BaseModel
from app.models.prediction_input import PredictionInput
from typing import List


class BatchPredictionInput(BaseModel):
    data: List[PredictionInput]

