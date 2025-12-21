from pydantic import BaseModel, ConfigDict
from typing import List

class BatchPredictionOutput(BaseModel):
    model_config = ConfigDict(protected_namespaces=())
    predictions: List[float]
    model_name: str
    model_version: str
    