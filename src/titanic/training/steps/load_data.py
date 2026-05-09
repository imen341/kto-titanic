import logging
import os
from pathlib import Path
import tempfile

import boto3
import mlflow
import pandas as pd

ARTIFACT_PATH = "path_output"


def load_data(path: str) -> str:
    logging.warning(f"load_data on path : {path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir, "data.csv")
        logging.warning(f"to path : {local_path}")

        s3_client = boto3.client(
            "s3",
            endpoint_url=os.environ.get("MLFLOW_S3_ENDPOINT_URL"),
            aws_access_key_id=os.environ.get("AWS_ACCESS_KEY_ID"),
            aws_secret_access_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        )

        try:
            s3_client.download_file("kto-titanic", path, str(local_path))
        except Exception:
            fallback_path = Path("/projects/kto-titanic/data/all_titanic.csv")
            if fallback_path.exists():
                local_path.write_bytes(fallback_path.read_bytes())
            else:
                raise

        pd.read_csv(local_path)
        mlflow.log_artifact(str(local_path), ARTIFACT_PATH)

    return f"{ARTIFACT_PATH}/data.csv"
