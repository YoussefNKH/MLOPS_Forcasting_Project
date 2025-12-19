import os
from src.utils import load_data
from src.preprocessing import split_train_valid, prepare_features
from src.train.trainer import train
from src.evaluate import evaluate_model
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv
from config.config import common_params

load_dotenv()

def main():
    mlflow.set_tracking_uri("http://20.199.136.148:5001")
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
    
    df = load_data("data/CA_1_0.pkl")
    df_train, df_valid = split_train_valid(df)
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