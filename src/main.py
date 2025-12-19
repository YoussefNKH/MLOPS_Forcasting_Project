from src.utils import load_data
from src.preprocessing import split_train_valid, prepare_features
from src.train.trainer import train
from src.evaluate import evaluate_model
from src.utils import save_model
import mlflow
import mlflow.sklearn
from dotenv import load_dotenv

load_dotenv()


def main():
    mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(os.getenv("MLFLOW_EXPERIMENT_NAME"))
    df = load_data("data/CA_1_0.pkl")
    df_train, df_valid = split_train_valid(df)

    X_train, X_valid, y_train, y_valid = prepare_features(df_train, df_valid)

    common_params = {
        "learning_rate": 0.05,
        "n_estimators": 1000,
        "random_state": 42
    }

    for model_name in ["lgbm", "catboost", "xgboost","svm"]:
        print(f"\nTraining {model_name}...")
        model = train(
            model_name,
            X_train, y_train,
            X_valid, y_valid,
            common_params
        )

        metrics = evaluate_model(model, X_valid, y_valid)
        print(metrics)

        save_model(model, f"models/{model_name}.pkl")


if __name__ == "__main__":
    main()
