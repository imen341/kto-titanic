import logging

import joblib
import mlflow
import pandas as pd
from mlflow.models import infer_signature
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, median_absolute_error

client = mlflow.MlflowClient()


def validate(model_path: str, x_test_path: str, y_test_path: str) -> None:
    logging.warning(f"validate {model_path}")

    model = joblib.load(
        client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=model_path)
    )

    x_test = pd.read_csv(
        client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=x_test_path),
        index_col=False,
    )
    y_test = pd.read_csv(
        client.download_artifacts(run_id=mlflow.active_run().info.run_id, path=y_test_path),
        index_col=False,
    )

    x_test = pd.get_dummies(x_test)

    if y_test.shape[1] == 1:
        y_test = y_test.iloc[:, 0]

    y_pred = model.predict(x_test)

    mlflow.log_metric("mse", mean_squared_error(y_test, y_pred))
    mlflow.log_metric("mae", mean_absolute_error(y_test, y_pred))
    mlflow.log_metric("r2", r2_score(y_test, y_pred))
    mlflow.log_metric("medae", median_absolute_error(y_test, y_pred))

    feature_importance = {
        name: float(importance)
        for name, importance in zip(x_test.columns.tolist(), model.feature_importances_, strict=False)
    }
    mlflow.log_dict(feature_importance, "feature_importance.json")

    model_info = mlflow.sklearn.log_model(
        model,
        name="model_final",
        signature=infer_signature(x_test, y_pred),
        input_example=x_test.head(10),
    )

    try:
        mlflow.register_model(model_info.model_uri, "model_registered")
    except Exception as e:
        logging.error(f"Erreur registry: {e}")
