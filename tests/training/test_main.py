from unittest.mock import patch

from titanic.training.main import workflow


def test_workflow_runs_all_steps():
    with (
        patch("titanic.training.main.load_data", return_value="path_output/data.csv") as mock_load_data,
        patch(
            "titanic.training.main.split_train_test",
            return_value=("xtrain/xtrain.csv", "xtest/xtest.csv", "ytrain/ytrain.csv", "ytest/ytest.csv"),
        ) as mock_split,
        patch("titanic.training.main.train", return_value="model_trained/model.joblib") as mock_train,
        patch("titanic.training.main.validate") as mock_validate,
        patch("mlflow.start_run"),
        patch("mlflow.log_param"),
    ):
        workflow("all_titanic.csv", 100, 10, 42)

        mock_load_data.assert_called_once_with("all_titanic.csv")
        mock_split.assert_called_once_with("path_output/data.csv")
        mock_train.assert_called_once_with("xtrain/xtrain.csv", "ytrain/ytrain.csv", 100, 10, 42)
        mock_validate.assert_called_once_with("model_trained/model.joblib", "xtest/xtest.csv", "ytest/ytest.csv")
