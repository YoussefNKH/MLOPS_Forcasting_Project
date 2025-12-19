import os
from src.utils import load_data
from src.preprocessing import split_train_test, prepare_features
from src.train.trainer import train
from src.evaluate import evaluate_model
from dotenv import load_dotenv
from src.config.config import common_params
import json
from datetime import datetime

load_dotenv()

def main():
    # Load and prepare data
    df = load_data("data/CA_1_0.pkl")
    df_train, df_valid = split_train_test(df)
    X_train, X_valid, y_train, y_valid = prepare_features(df_train, df_valid)
    
    # Track best model
    best_score = float('inf')  # Lower is better for combined metric
    best_model = None
    best_model_name = None
    best_metrics = None
    
    results = []
    
    for model_name in ["lgbm", "catboost", "xgboost"]:
        print(f"\n{'='*50}")
        print(f"Training {model_name}...")
        print(f"{'='*50}")
        
        # Train model
        model = train(
            model_name,
            X_train, y_train,
            X_valid, y_valid,
            common_params
        )
        
        # Evaluate model
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
        
        # Print metrics
        print(f"\n{model_name} Results:")
        print(f"  RMSE: {metrics['rmse']:.4f}")
        print(f"  MAE: {metrics['mae']:.4f}")
        print(f"  MSE: {metrics['mse']:.4f}")
        print(f"  R2: {metrics['r2']:.4f}")
        print(f"  Combined Metric: {combined_metric:.4f}")
        
        # Store results
        result = {
            "model_name": model_name,
            "metrics": metrics,
            "params": common_params,
            "timestamp": datetime.now().isoformat()
        }
        results.append(result)
        
        # Track best model
        if combined_metric < best_score:
            best_score = combined_metric
            best_model = model
            best_model_name = model_name
            best_metrics = metrics
    
    # Print best model summary
    print(f"\n{'='*50}")
    print("BEST MODEL SUMMARY")
    print(f"{'='*50}")
    print(f"Best Model: {best_model_name}")
    print(f"Best Combined Metric: {best_score:.4f}")
    print(f"\nBest Model Metrics:")
    for metric, value in best_metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # Save results to JSON file
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)
    
    results_file = os.path.join(results_dir, f"training_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(results_file, 'w') as f:
        json.dump({
            "all_results": results,
            "best_model": {
                "name": best_model_name,
                "metrics": best_metrics,
                "combined_score": best_score
            }
        }, f, indent=2)
    
    print(f"\nResults saved to: {results_file}")
    
    # Optionally save the best model (using joblib or pickle)
    try:
        import joblib
        model_file = os.path.join(results_dir, f"best_model_{best_model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl")
        joblib.dump(best_model, model_file)
        print(f"Best model saved to: {model_file}")
    except ImportError:
        print("Note: Install joblib to save models (pip install joblib)")
    
    return best_model, best_metrics

if __name__ == "__main__":
    main()