from .lgbm import train_lgbm
from .catboost import train_catboost
from .xgboost import train_xgboost



def train(model_name, X_train, y_train, X_valid, y_valid, common_params=None):
    if model_name == "lgbm":
        return train_lgbm(X_train, y_train, X_valid, y_valid, common_params)

    elif model_name == "catboost":
        return train_catboost(X_train, y_train, X_valid, y_valid)

    elif model_name == "xgboost":
        return train_xgboost(X_train, y_train, X_valid, y_valid, common_params)
    else:
        raise ValueError(f"Unknown model: {model_name}")