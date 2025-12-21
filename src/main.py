import os
from src.utils import load_data , get_latest_data_file
from src.preprocessing import split_train_test, prepare_features
from src.train.trainer import train
from src.evaluate import evaluate_model
from src.config.config import common_params , MLFLOW_TRACKING_URI_PORT , MLFLOW_EXPERIMENT_NAME
import json
from datetime import datetime
import mlflow
import mlflow.sklearn


def main():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI_PORT)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_NAME)
    try:
        data_path = get_latest_data_file(data_dir="data")
        print(f"--- Processing latest data file: {data_path} ---")
        df = load_data(data_path)
    except Exception as e:
        print(f"Error loading data: {e}")
        return 
   # df = load_data("data/CA_1_0.pkl")
    df_train, df_valid = split_train_test(df)
    X_train, X_valid, y_train, y_valid = prepare_features(df_train, df_valid)
    
    # Track best model
    best_score = float('inf')  # Lower is better for combined metric
    best_run_id = None
    best_model_name = None
    
    for model_name in ["lgbm", "catboost", "xgboost"]:
        with mlflow.start_run(run_name=model_name) as run:
            model = train(
                model_name,
                X_train, y_train,
                X_valid, y_valid,
                common_params
            )
            
            metrics = evaluate_model(model, X_valid, y_valid)
            
            # Calculate combined metric (weighted average)
            combined_metric = (
                0.4 * metrics["rmse"] + 
                0.3 * metrics["mae"] + 
                0.2 * metrics["mse"] + 
                0.1 * (1 - metrics["r2"])
            )
            
            # Add combined metric to metrics dict
            metrics["combined_metric"] = combined_metric
            
            # Log all metrics
            for k, v in metrics.items():
                mlflow.log_metric(k, v)
            
            # Log parameters
            mlflow.log_param("model_name", model_name)
            for k, v in common_params.items():
                mlflow.log_param(k, v)
            
            # Log model to current run
            mlflow.sklearn.log_model(
                model,
                artifact_path="model"
            )
            
            print(f"{model_name} - Combined Metric: {combined_metric:.4f}")
            
            # Track best model
            if combined_metric < best_score:
                best_score = combined_metric
                best_run_id = run.info.run_id
                best_model_name = model_name
    
    # Register only the best model to Model Registry
    print(f"\nBest Model: {best_model_name}")
    print(f"Best Combined Metric: {best_score:.4f}")
    print(f"Best Run ID: {best_run_id}")
    
    model_uri = f"runs:/{best_run_id}/model"
    mlflow.register_model(
        model_uri=model_uri,
        name="BestRegressionModel"
    )
    
    print(f"Best model registered to Model Registry as 'BestRegressionModel'")

if __name__ == "__main__":
    main()