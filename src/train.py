import pandas as pd 
import pickle as pkl 
import lightgbm as lgb 
from lightgbm import LGBMRegressor 
from catboost import CatBoostRegressor
from xgboost import XGBRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

# Load data
with open('data/CA_1_0.pkl', 'rb') as f:
    df_train = pkl.load(f)

with open('data/CA_1_1.pkl', 'rb') as f:
    df_valid = pkl.load(f)

# Prepare features and targets
y_train = df_train['sold']
y_valid = df_valid['sold']
X_train = df_train.drop('sold', axis=1)
X_valid = df_valid.drop('sold', axis=1)

# Common parameters
common_params = {
    'n_estimators': 1000,
    'learning_rate': 0.3,
    'max_depth': 8,
    'random_state': 42
}

# 1. LightGBM Regressor (sklearn API for consistency)
lgbm_model = LGBMRegressor(
    **common_params,
    num_leaves=50,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=300,
    n_jobs=-1
)
lgbm_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_valid, y_valid)],
    callbacks=[
        lgb.early_stopping(10, verbose=True),
        lgb.log_evaluation(5)
    ]
)
pkl.dump(lgbm_model, open('models/lgbm.pkl', 'wb'))

# 2. CatBoost Regressor
cat_model = CatBoostRegressor(
    iterations=1000,
    learning_rate=0.3,
    depth=8,
    loss_function='RMSE',
    early_stopping_rounds=10,
    verbose=5,
    random_state=42
)
cat_model.fit(X_train, y_train, eval_set=(X_valid, y_valid))
pkl.dump(cat_model, open('models/catboost.pkl', 'wb'))

# 3. XGBoost Regressor
xgb_model = XGBRegressor(
    **common_params,
    subsample=0.8,
    colsample_bytree=0.8,
    n_jobs=-1,
    early_stopping_rounds=10
)
xgb_model.fit(
    X_train, y_train, 
    eval_set=[(X_valid, y_valid)], 
    verbose=5
)
pkl.dump(xgb_model, open('models/xgboost.pkl', 'wb'))

# 4. SVM Regressor (scale features first!)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_valid_scaled = scaler.transform(X_valid)

svm_model = SVR(kernel='rbf', C=1.0, gamma='scale')
svm_model.fit(X_train_scaled, y_train)
pkl.dump((svm_model, scaler), open('models/svm.pkl', 'wb'))  # Save scaler too!