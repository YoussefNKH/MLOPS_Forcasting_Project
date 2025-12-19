import lightgbm as lgb
from lightgbm import LGBMRegressor

def train_lgbm(X_train, y_train, X_valid, y_valid, common_params):
    model = LGBMRegressor(
        **common_params,
        num_leaves=50,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=300,
        n_jobs=-1
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_valid, y_valid)],
        callbacks=[
            lgb.early_stopping(10, verbose=True),
            lgb.log_evaluation(5)
        ]
    )
    return model