from fastapi import HTTPException
import mlflow
import mlflow.pyfunc

def load_best_model_from_mlflow(experiment_name: str = "sales_forecasting", model_name: str = "BestRegressionModel"):

    try:
        # Set MLflow tracking URI
        mlflow.set_tracking_uri("http://20.199.136.148:5001")
        
        # Option 1: Load from Model Registry (Recommended)
        try:
            # Load the latest version of the model
            model_uri = f"models:/{model_name}/latest"
            loaded_model = mlflow.pyfunc.load_model(model_uri)
            
            # Get model version info
            client = mlflow.tracking.MlflowClient()
            latest_versions = client.get_latest_versions(model_name, stages=["None", "Production", "Staging"])
            
            model_info = {}
            if latest_versions:
                latest_version = max(latest_versions, key=lambda x: int(x.version))
                model_info = {
                    "name": model_name,
                    "version": latest_version.version,
                    "stage": latest_version.current_stage,
                    "run_id": latest_version.run_id
                }
            
            return loaded_model, model_info
            
        except Exception as registry_error:
            # Option 2: Load best model from experiment runs
            if experiment_name:
                # Search for the best run based on metrics
                experiment = mlflow.get_experiment_by_name(experiment_name)
                if not experiment:
                    raise ValueError(f"Experiment '{experiment_name}' not found")
                
                # Get the best run (lowest RMSE or your preferred metric)
                runs = mlflow.search_runs(
                    experiment_ids=[experiment.experiment_id],
                    order_by=["metrics.rmse ASC"],  # Change metric as needed
                    max_results=1
                )
                
                if runs.empty:
                    raise ValueError(f"No runs found in experiment '{experiment_name}'")
                
                best_run_id = runs.iloc[0].run_id
                model_uri = f"runs:/{best_run_id}/model"
                loaded_model = mlflow.pyfunc.load_model(model_uri)
                
                model_info = {
                    "name": "best_model_from_experiment",
                    "version": "N/A",
                    "run_id": best_run_id,
                    "experiment": experiment_name
                }
                
                return loaded_model, model_info
            
            raise Exception("Could not load model from Registry or Experiments")
            
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to load model: {str(e)}")