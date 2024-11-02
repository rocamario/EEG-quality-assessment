import xgboost as xgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
from models.features_list import list_features

def standardize_data(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    columns_to_scale = list_features
    
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_val_scaled[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
    
    return X_train_scaled, X_val_scaled

# Train an XGBoost model
def train_xgboost_model(x_train, y_train):
    dtrain = xgb.DMatrix(data=x_train[list_features], label=y_train)
    
    params = {
        'objective': 'binary:logistic',
        'max_depth': 6,
        'eta': 0.3,
        'eval_metric': 'logloss'
    }
    
    model = xgb.train(params, dtrain, num_boost_round=100)
    return model

# Evaluate the model
def evaluate_model(model, x_val, y_val):
    dval = xgb.DMatrix(data=x_val[list_features])
    
    prediction = model.predict(dval)
    prediction = np.round(prediction)  # Convert probabilities to binary predictions
    
    f1 = f1_score(y_val, prediction)
    cohen_k = cohen_kappa_score(y_val, prediction)
    
    print(f"Cohen's Kappa: {cohen_k}")
    print(f"F1 Score: {f1}")