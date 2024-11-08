from tpot import TPOTClassifier
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler
from models.features_list import best_features

def standardize_data(X_train, X_val):
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    columns_to_scale = best_features
    
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_val_scaled[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
    
    return X_train_scaled, X_val_scaled


# Train an AutoML model
def train_automl_model(x_train, y_train):
    
    automl = TPOTClassifier(generations=5, population_size=50, verbosity=2, random_state=42)
    automl.fit(np.array(x_train[best_features]), y_train)

    return automl

# Evaluate the model
def evaluate_model(model, x_val, y_val):

    prediction = model.predict(np.array(x_val[best_features]))
    print("Cohen: ", cohen_kappa_score(prediction, y_val))
    print("F1 score: ", f1_score(y_val, prediction))