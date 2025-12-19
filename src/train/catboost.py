from catboost import CatBoostRegressor

def train_catboost(X_train, y_train, X_valid, y_valid):
    model = CatBoostRegressor(
        iterations=1000,
        learning_rate=0.3,
        depth=8,
        loss_function="RMSE",
        early_stopping_rounds=10,
        verbose=5,
        random_state=42
    )

    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
    return model