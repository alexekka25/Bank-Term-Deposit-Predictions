import argparse
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import pandas as pd

def train_model(X_train, y_train, n_estimators, max_depth):
    rf_clf = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, random_state=42)
    rf_clf.fit(X_train, y_train)
    return rf_clf

def evaluate_model(model, X_val, y_val):
    predictions = model.predict(X_val)
    report = classification_report(y_val, predictions)
    matrix = confusion_matrix(y_val, predictions)
    return report, matrix

def log_model_with_mlflow(model, report, matrix, X_sample, n_estimators, max_depth):
    with mlflow.start_run():
        # Infer the model signature
        signature = mlflow.models.infer_signature(X_sample, model.predict(X_sample))

        # Log parameters, metrics, and model to MLflow
        mlflow.sklearn.log_model(model, "model", signature=signature)
        mlflow.log_param("model_type", "RandomForest")
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)
        mlflow.log_metric("accuracy", model.score(X_val, y_val))
        mlflow.log_text(report, "classification_report.txt")
        mlflow.log_text(str(matrix), "confusion_matrix.txt")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train a Random Forest model with different parameters.")
    parser.add_argument("--n_estimators", type=int, default=100, help="Number of trees in the forest.")
    parser.add_argument("--max_depth", type=int, default=None, help="Maximum depth of the trees.")
    args = parser.parse_args()

    X_train = pd.read_csv('X_train.csv')
    y_train = pd.read_csv('y_train.csv').values.ravel()
    X_val = pd.read_csv('X_val.csv')
    y_val = pd.read_csv('y_val.csv').values.ravel()
    
    # Ensure y_train and y_val are categorical
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_val = label_encoder.transform(y_val)
    
    # Create a sample input for signature inference
    X_sample = X_val.head(5)  # Using a small sample from validation data

    model = train_model(X_train, y_train, args.n_estimators, args.max_depth)
    report, matrix = evaluate_model(model, X_val, y_val)
    
    log_model_with_mlflow(model, report, matrix, X_sample, args.n_estimators, args.max_depth)
