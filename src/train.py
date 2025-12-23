import subprocess
import yaml
import pandas as pd
import mlflow
import joblib
import os
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score
from processing_data import process_data

def get_git_commit():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
    except Exception:
        return "unknown"

def main():
    # Start MLflow run
    mlflow.start_run(
        run_name=f"stroke-prediction-{get_git_commit()}",
        tags={"git_commit": get_git_commit()}
    )

    # Load configuration
    with open("config.yaml", "r") as f:
        full_config = yaml.load(f, Loader=yaml.FullLoader)
        data_config = full_config.get("data", {})
        train_config = full_config.get("train", {})

    mlflow.log_params(train_config)
    mlflow.log_params(data_config)

    # Load data
    df = pd.read_csv(data_config["path"])
    df = df.drop(columns=["id"])
    print(df.shape)

    # Process data
    X_train, X_val, y_train, y_val = process_data(df)

    # Train XGBoost model
    model = XGBClassifier(
        n_estimators=train_config["n_estimators"],
        learning_rate=train_config["learning_rate"],
        max_depth=train_config["max_depth"],
        gamma=train_config["gamma"],
        subsample=train_config["subsample"],
        colsample_bytree=train_config["colsample_bytree"],
        random_state=train_config["random_state"],
        objective=train_config["objective"]
    )
    model.fit(X_train, y_train)

    # Evaluate model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    f1 = f1_score(y_val, y_pred)
    print(f"Accuracy: {accuracy}")
    print(f"F1 Score: {f1}")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1", f1)

    # Save model
    os.makedirs("models", exist_ok=True) 
    model_path = os.path.join("models", "xgb_model.pkl")
    joblib.dump(model, model_path)
    print(f"Model saved to {model_path}")

    mlflow.end_run()

if __name__ == "__main__":
    main()
