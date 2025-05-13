import argparse
import inspect
import os
import sys
import logging
import yaml

from oligogym.data import *
from oligogym.features import *
from oligogym.metrics import *
from oligogym.models import *
import pandas as pd

class ConfigurationError(Exception):
    pass

class DatasetDownloaderError(Exception):
    pass

def load_yaml(file_path):
    with open(file_path, "r") as f:
        return yaml.safe_load(f)

def check_config(config):
    if (config["featurizer"] == "KMersCounts") and (
        config["model"] in ["CNN", "GRU", "CausalCNN"]
    ):
        raise ConfigurationError(
            "KMersCounts featurizer is not compatible with CNN, GRU, or CausalCNN models"
        )
    if config["model"] == "CNN":
        if (config["model_args"]["depth"] == 2) and (
            config["model_args"]["kernel_size"] > 5
        ):
            raise ConfigurationError("CNN with depth 2 requires kernel_size <= 5")

def download_data_with_retries(config, max_retries=100):
    for attempt in range(max_retries):
        try:
            downloader = DatasetDownloader()
            data = downloader.download(config["dataset"])
            return data
        except Exception as e:
            logging.error(f"Attempt {attempt} failed with error: {e}")
            continue
    raise DatasetDownloaderError("Failed to download data after multiple attempts")

def prepare_data(data, split_strategy="random"):
    X_train, X_test, y_train, y_test, train_indices, test_indices = data.split(split_strategy, return_index=True)
    return X_train, X_test, y_train, y_test, train_indices, test_indices

def prepare_data_fold(data, k):
    X = data.x
    y = data.y
    n = len(X)
    indices = np.arange(n)
    np.random.shuffle(indices)
    fold_size = n // 5
    test_indices = indices[k * fold_size : (k + 1) * fold_size]
    train_indices = np.concatenate(
        [indices[: k * fold_size], indices[(k + 1) * fold_size :]]
    )
    X_train, X_test = X[train_indices], X[test_indices]
    y_train, y_test = y[train_indices], y[test_indices]
    return X_train, X_test, y_train, y_test, train_indices, test_indices


def prepare_featurizer(config):
    if config["featurizer"] == "OneHotEncoder":
        return OneHotEncoder(**config["featurizer_args"])
    elif config["featurizer"] == "KMersCounts":
        return KMersCounts(**config["featurizer_args"])
    elif config["featurizer"] == "Thermodynamics":
        return Thermodynamics(config["featurizer_args"])
    elif config["featurizer"] == "TargetDescriptors":
        return TargetDescriptors(**config["featurizer_args"])


def prepare_model(config):
    model_args = config["model_args"]
    valid_args = inspect.signature(eval(config["model"])).parameters.keys()
    model_args = {k: v for k, v in model_args.items() if k in valid_args}

    if config["model"] == "NearestNeighborsModel":
        return NearestNeighborsModel(**model_args)
    elif config["model"] == "LinearModel":
        return LinearModel(**model_args)
    elif config["model"] == "RandomForestModel":
        return RandomForestModel(**model_args)
    elif config["model"] == "GaussianProcessModel":
        return GaussianProcessModel(**model_args)
    elif config["model"] == "XGBoostModel":
        return XGBoostModel(**model_args)
    elif config["model"] == "TabPFNModel":
        return TabPFNModel(**model_args)
    elif config["model"] == "CNN":
        return CNN(**model_args)
    elif config["model"] == "MLP":
        return MLP(**model_args)
    elif config["model"] == "GRU":
        return GRU(**model_args)
    elif config["model"] == "CausalCNN":
        return CausalCNN(**model_args)


def featurize(X_train, X_test, featurizer, config):
    X_train = featurizer.fit_transform(X_train)
    X_test = featurizer.transform(X_test)
    if (
        config["model"]
        in [
            "LinearModel",
            "RandomForestModel",
            "GaussianProcessModel",
            "XGBoostModel",
            "NearestNeighborsModel",
            "TabPFNModel",
            "MLP"
        ]
        and len(X_train.shape) == 3
    ):
        X_train = X_train.reshape(X_train.shape[0], -1)
        X_test = X_test.reshape(X_test.shape[0], -1)
    return X_train, X_test


def predict(model, X_train, X_test, y_train):
    model.fit(X_train, y_train)
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    return y_pred_train, y_pred_test


def main(config):
    results_dir = os.path.dirname(config)
    config = load_yaml(config)
    check_config(config)
    os.chdir(results_dir)

    if config['cross_validation'] == "random":
        regression_metrics_train_folds = []
        regression_metrics_test_folds = []
        train_indices_folds = []
        test_indices_folds = []
        for k in range(5):
            #data and traning
            data = download_data_with_retries(config)
            X_train, X_test, y_train, y_test, train_indices, test_indices = prepare_data_fold(data, k)
            featurizer = prepare_featurizer(config)
            X_train, X_test = featurize(X_train, X_test, featurizer, config)
            config["model_args"]["input_dim"] = X_train.shape[-1]
            model = prepare_model(config)
            y_pred_train, y_pred_test = predict(model, X_train, X_test, y_train)

            #collect metrics
            regression_metrics_train = regression_metrics(
                y_train.squeeze(), y_pred_train.squeeze()
            )
            regression_metrics_test = regression_metrics(
                y_test.squeeze(), y_pred_test.squeeze()
            )
            regression_metrics_train = pd.DataFrame(
                regression_metrics_train, index=[0]
            ).assign(fold=k)
            regression_metrics_test = pd.DataFrame(
                regression_metrics_test, index=[0]
            ).assign(fold=k)
            regression_metrics_train_folds.append(regression_metrics_train)
            regression_metrics_test_folds.append(regression_metrics_test)

            #store indices
            train_indices_df = pd.DataFrame({f"fold_{k}": train_indices}, index=range(len(train_indices)))
            test_indices_df = pd.DataFrame({f"fold_{k}": test_indices}, index=range(len(test_indices)))
            train_indices_folds.append(train_indices_df)
            test_indices_folds.append(test_indices_df)

        # Concatenate all folds
        train_indices_all = pd.concat(train_indices_folds, axis=1)
        test_indices_all = pd.concat(test_indices_folds, axis=1)
        train_indices_all.to_csv(
            os.path.join(results_dir, "train_indices.csv"), index=False
        )
        test_indices_all.to_csv(
            os.path.join(results_dir, "test_indices.csv"), index=False
        )
        
        #store metrics
        regression_metrics_train = pd.concat(regression_metrics_train_folds)
        regression_metrics_test = pd.concat(regression_metrics_test_folds)
        regression_metrics_train.to_csv(
            os.path.join(results_dir, "regression_metrics_train.csv"), index=False
        )
        regression_metrics_test.to_csv(
            os.path.join(results_dir, "regression_metrics_test.csv"), index=False
        )
        return None
    
    elif config['cross_validation'] == "nucleobase":
        regression_metrics_train_folds = []
        regression_metrics_test_folds = []
        train_indices_folds = []
        test_indices_folds = []
        for k in range(5):
            #data and traning
            data = download_data_with_retries(config)
            X_train, X_test, y_train, y_test, train_indices, test_indices = prepare_data(data, split_strategy="nucleobase")
            featurizer = prepare_featurizer(config)
            X_train, X_test = featurize(X_train, X_test, featurizer, config)
            config["model_args"]["input_dim"] = X_train.shape[-1]
            model = prepare_model(config)
            y_pred_train, y_pred_test = predict(model, X_train, X_test, y_train)

            #collect metrics
            regression_metrics_train = regression_metrics(
                y_train.squeeze(), y_pred_train.squeeze()
            )
            regression_metrics_test = regression_metrics(
                y_test.squeeze(), y_pred_test.squeeze()
            )
            regression_metrics_train = pd.DataFrame(
                regression_metrics_train, index=[0]
            ).assign(fold=k)
            regression_metrics_test = pd.DataFrame(
                regression_metrics_test, index=[0]
            ).assign(fold=k)
            regression_metrics_train_folds.append(regression_metrics_train)
            regression_metrics_test_folds.append(regression_metrics_test)

            #store indices
            train_indices_df = pd.DataFrame({f"fold_{k}": train_indices}, index=range(len(train_indices)))
            test_indices_df = pd.DataFrame({f"fold_{k}": test_indices}, index=range(len(test_indices)))
            train_indices_folds.append(train_indices_df)
            test_indices_folds.append(test_indices_df)
        
        # Concatenate all folds
        train_indices_all = pd.concat(train_indices_folds, axis=1)
        test_indices_all = pd.concat(test_indices_folds, axis=1)
        train_indices_all.to_csv(
            os.path.join(results_dir, "train_indices.csv"), index=False
        )
        test_indices_all.to_csv(
            os.path.join(results_dir, "test_indices.csv"), index=False
        )

        #store metrics
        regression_metrics_train = pd.concat(regression_metrics_train_folds)
        regression_metrics_test = pd.concat(regression_metrics_test_folds)
        regression_metrics_train.to_csv(
            os.path.join(results_dir, "regression_metrics_train.csv"), index=False
        )
        regression_metrics_test.to_csv(
            os.path.join(results_dir, "regression_metrics_test.csv"), index=False
        )
        return None

    elif config['cross_validation'] == "none":
        data = download_data_with_retries(config)
        X_train, X_test, y_train, y_test, train_indices, test_indices = prepare_data(data)
        featurizer = prepare_featurizer(config)
        X_train, X_test = featurize(X_train, X_test, featurizer, config)
        config["model_args"]["input_dim"] = X_train.shape[-1]
        model = prepare_model(config)
        y_pred_train, y_pred_test = predict(model, X_train, X_test, y_train)
        regression_metrics_train = regression_metrics(
            y_train.squeeze(), y_pred_train.squeeze()
        )
        regression_metrics_test = regression_metrics(
            y_test.squeeze(), y_pred_test.squeeze()
        )
        pd.DataFrame(regression_metrics_test, index=[0]).to_csv(
            os.path.join(results_dir, "regression_metrics_test.csv"), index=False
        )
        pd.DataFrame(regression_metrics_train, index=[0]).to_csv(
            os.path.join(results_dir, "regression_metrics_train.csv"), index=False
        )
        pd.DataFrame({"train_indices": train_indices}).to_csv(
            os.path.join(results_dir, f"train_indices.csv"), index=False
        )
        pd.DataFrame({"test_indices": test_indices}).to_csv(
            os.path.join(results_dir, f"test_indices.csv"), index=False
        )
        return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate configurations for training models"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Configuration file for generating runs",
    )
    args = parser.parse_args()

    config = args.config
    main(config)
