from xgboost import XGBRegressor

def train_xgboost(X_train, y_train, X_valid, y_valid, common_params):
    model = XGBRegressor(
        **common_params,
        subsample=0.8,
        colsample_bytree=0.8,
        n_jobs=-1,
        early_stopping_rounds=10
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_valid, y_valid)],
        verbose=5
    )
    return model
