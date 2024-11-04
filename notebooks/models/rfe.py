from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.feature_selection import SequentialFeatureSelector

def train_knn_with_sfs(x_train, y_train, num_features=10, n_neighbors=11):
    # Initialize the KNN model
    knn_model = KNeighborsClassifier(n_neighbors=n_neighbors)

    # Initialize Sequential Feature Selector with the KNN model and the desired number of features
    sfs = SequentialFeatureSelector(knn_model, n_features_to_select=num_features, direction='forward')

    # Fit SFS
    sfs.fit(x_train, y_train)

    # Get the selected features
    selected_features = x_train.columns[sfs.get_support()]
    print("Selected features:", selected_features)

    # Train the KNN model with the selected features
    knn_model.fit(x_train[selected_features], y_train)

    return knn_model, selected_features

def evaluate_model(model, x_test, y_test):
    # Make predictions
    y_pred = model.predict(x_test)

    # Calculate Cohen's kappa score
    kappa_score = cohen_kappa_score(y_test, y_pred)
    print("Cohen's kappa score:", kappa_score)

    # Calculate F1 score
    f1 = f1_score(y_test, y_pred, average='weighted')
    print("F1 score:", f1)