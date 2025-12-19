import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def evaluate_model(model, X_valid, y_valid):
    y_pred = model.predict(X_valid)

    rmse = np.sqrt(mean_squared_error(y_valid, y_pred))
    mse = mean_squared_error(y_valid, y_pred)
    mae = mean_absolute_error(y_valid, y_pred)
    r2 = r2_score(y_valid, y_pred)

    return {
        "rmse": rmse,
        "mse": mse,
        "mae": mae,
        "r2": r2
    }