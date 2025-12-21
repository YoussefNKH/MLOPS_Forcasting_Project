from dotenv import load_dotenv
import os
load_dotenv()
common_params = {
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "random_state": 42
    }
MLFLOW_TRACKING_URI_PORT = os.getenv("MLFLOW_TRACKING_URI_PORT")
MLFLOW_EXPERIMENT_NAME = os.getenv("MLFLOW_EXPERIMENT_NAME")

