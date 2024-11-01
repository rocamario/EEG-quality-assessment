from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.preprocessing import StandardScaler

def standardize_data(X_train, X_val):
    
    scaler = StandardScaler()
    X_train_scaled = X_train.copy()
    X_val_scaled = X_val.copy()
    
    columns_to_scale = ["amplitude", "mean", "max", "min", "stdev",
                        "delta_power", "theta_power", "alpha_power",
                        "beta_power", "gamma_power"]
    
    X_train_scaled[columns_to_scale] = scaler.fit_transform(X_train[columns_to_scale])
    X_val_scaled[columns_to_scale] = scaler.transform(X_val[columns_to_scale])
    
    return X_train_scaled, X_val_scaled


# Train a baseline model
def train_baseline_model(x_train, y_train):
    
    neigh = KNeighborsClassifier(n_neighbors=8)
    neigh.fit(np.array(x_train[["amplitude", "mean", "max", "min", "stdev",
                                "delta_power", "theta_power", "alpha_power",
                                "beta_power", "gamma_power"]]), y_train)

    return neigh

# Evaluate the model
def evaluate_model (model, x_val, y_val):

    prediction = model.predict(np.array(x_val[["amplitude", "mean", "max", "min", "stdev",
                                               "delta_power", "theta_power", "alpha_power",
                                               "beta_power", "gamma_power"]]))
    print(cohen_kappa_score(prediction, y_val))
    print(f1_score(y_val, prediction))