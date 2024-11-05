import lightgbm as lgb
import numpy as np
from sklearn.metrics import f1_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import cohen_kappa_score
from models.features_list import collinear_features

def standardize_data(X_train, X_val):
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    columns_to_scale = collinear_features
    
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_val_scaled[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
    
    return X_train_scaled, X_val_scaled

# Train a LightGBM model
def train_lgbm_model(x_train, y_train):
    train_data = lgb.Dataset(data=x_train[collinear_features], label=y_train)
    
    params = {
        'objective': 'binary',
        'max_depth': 6,
        'learning_rate': 0.3,
        'num_leaves': 31,
        'force_row_wise': True,
        'metric': 'binary_logloss'
    }
    
    model = lgb.train(params, train_data, num_boost_round=100)
    return model

# Evaluate the model
def evaluate_model(model, x_val, y_val):
    prediction = model.predict(x_val[collinear_features])
    prediction = np.round(prediction)  # Convert probabilities to binary predictions
    
    f1 = f1_score(y_val, prediction)
    cohen_k = cohen_kappa_score(y_val, prediction)
    
    print(f"Cohen's Kappa: {cohen_k}")
    print(f"F1 Score: {f1}")
