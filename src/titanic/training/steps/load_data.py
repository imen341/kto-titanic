import logging
from pathlib import Path
import tempfile

import mlflow
import pandas as pd

ARTIFACT_PATH = "path_output"


def load_data(path: str) -> str:
    logging.warning(f"load_data on path : {path}")

    with tempfile.TemporaryDirectory() as tmp_dir:
        local_path = Path(tmp_dir, "data.csv")
        logging.warning(f"to path : {local_path}")

        source_path = Path("/projects/kto-titanic/data/all_titanic.csv")
        local_path.write_bytes(source_path.read_bytes())

        pd.read_csv(local_path)
        mlflow.log_artifact(str(local_path), ARTIFACT_PATH)

    return f"{ARTIFACT_PATH}/data.csv"
