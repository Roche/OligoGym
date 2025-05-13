import numpy as np
from scipy import stats
from sklearn import metrics
from sklearn.preprocessing import label_binarize
import pandas as pd
from typing import Union


def r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate R^2 (coefficient of determination) regression score.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: R^2 score.
    """
    return metrics.r2_score(y_true, y_pred)


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate root mean squared error regression loss.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Root mean squared error.
    """
    mse = np.mean((y_true - y_pred) ** 2)
    rmse = np.sqrt(mse)
    return rmse


def mean_absolute_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate mean absolute error regression loss.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Mean absolute error.
    """
    return metrics.mean_absolute_error(y_true, y_pred)


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Pearson correlation coefficient between two arrays of numbers.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Pearson correlation coefficient.
    """
    return stats.pearsonr(y_true, y_pred)[0]


def spearman_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate the Spearman correlation coefficient between two arrays of numbers.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Spearman correlation coefficient.
    """
    return stats.spearmanr(y_true, y_pred)[0]


def auc_roc(
    y_true: np.ndarray, y_pred: np.ndarray, multi_class: str = "raise"
) -> float:
    """
    Calculate area under curve for receiver operating characteristic.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        multi_class (str): Determines the type of averaging performed on the data.

    Returns:
        float: AUC-ROC score.
    """
    if np.all(np.mod(y_pred, 1) == 0):
        raise ValueError(
            "y_pred contains discrete class labels, expected class probabilities."
        )

    if multi_class == "raise":
        return metrics.roc_auc_score(y_true, y_pred, multi_class=multi_class)
    else:
        return metrics.roc_auc_score(
            y_true, y_pred, multi_class=multi_class, average="macro"
        )


def auc_prc(
    y_true: np.ndarray, y_pred: np.ndarray, multi_class: str = "raise"
) -> float:
    """
    Calculate area under curve for precision recall curve.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        multi_class (str): Determines the type of averaging performed on the data.

    Returns:
        float: AUC-PRC score.
    """
    if np.all(np.mod(y_pred, 1) == 0):
        raise ValueError(
            "y_pred contains discrete class labels, expected class probabilities."
        )

    if multi_class == "raise":
        return metrics.average_precision_score(y_true, y_pred)
    else:
        if y_true.ndim == 1:
            y_true_bin = label_binarize(y_true, classes=np.unique(y_true))
        else:
            y_true_bin = y_true

        if y_pred.ndim == 1:
            y_pred_bin = label_binarize(y_pred, classes=np.unique(y_true))
        else:
            y_pred_bin = y_pred

        return metrics.average_precision_score(y_true_bin, y_pred_bin, average="macro")


def accuracy(y_true: np.ndarray, y_pred: np.ndarray, warning=False) -> float:
    """
    Calculate accuracy.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Accuracy score.
    """
    if (
        y_pred.ndim == 1
        and np.all((y_pred >= 0) & (y_pred <= 1))
        and not np.all(np.mod(y_pred, 1) == 0)
    ):
        y_pred = (y_pred >= 0.5).astype(int)
        if warning:
            print("Converted binary class probabilities to discrete labels.")
    elif (
        y_pred.ndim > 1
        and np.all((y_pred >= 0) & (y_pred <= 1))
        and not np.all(np.mod(y_pred, 1) == 0)
    ):
        y_pred = np.argmax(y_pred, axis=1)
        if warning:
            print("Converted multi-class probabilities to discrete labels.")

    return metrics.accuracy_score(y_true, y_pred)


def precision(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary", warning=False
) -> float:
    """
    Calculate precision.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        average (str): Type of averaging performed on the data.

    Returns:
        float: Precision score.
    """
    if (
        y_pred.ndim == 1
        and np.all((y_pred >= 0) & (y_pred <= 1))
        and not np.all(np.mod(y_pred, 1) == 0)
    ):
        y_pred = (y_pred >= 0.5).astype(int)
        if warning:
            print("Converted binary class probabilities to discrete labels.")
    elif (
        y_pred.ndim > 1
        and np.all((y_pred >= 0) & (y_pred <= 1))
        and not np.all(np.mod(y_pred, 1) == 0)
    ):
        y_pred = np.argmax(y_pred, axis=1)
        if warning:
            print("Converted multi-class probabilities to discrete labels.")

    return metrics.precision_score(y_true, y_pred, average=average)


def recall(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary", warning=False
) -> float:
    """
    Calculate recall.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        average (str): Type of averaging performed on the data.

    Returns:
        float: Recall score.
    """
    if (
        y_pred.ndim == 1
        and np.all((y_pred >= 0) & (y_pred <= 1))
        and not np.all(np.mod(y_pred, 1) == 0)
    ):
        y_pred = (y_pred >= 0.5).astype(int)
        if warning:
            print("Converted binary class probabilities to discrete labels.")
    elif (
        y_pred.ndim > 1
        and np.all((y_pred >= 0) & (y_pred <= 1))
        and not np.all(np.mod(y_pred, 1) == 0)
    ):
        y_pred = np.argmax(y_pred, axis=1)
        if warning:
            print("Converted multi-class probabilities to discrete labels.")

    return metrics.recall_score(y_true, y_pred, average=average)


def f1_score(
    y_true: np.ndarray, y_pred: np.ndarray, average: str = "binary", warning=False
) -> float:
    """
    Calculate F1 score.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        average (str): Type of averaging performed on the data.

    Returns:
        float: F1 score.
    """
    if (
        y_pred.ndim == 1
        and np.all((y_pred >= 0) & (y_pred <= 1))
        and not np.all(np.mod(y_pred, 1) == 0)
    ):
        y_pred = (y_pred >= 0.5).astype(int)
        if warning:
            print("Converted binary class probabilities to discrete labels.")
    elif (
        y_pred.ndim > 1
        and np.all((y_pred >= 0) & (y_pred <= 1))
        and not np.all(np.mod(y_pred, 1) == 0)
    ):
        y_pred = np.argmax(y_pred, axis=1)
        if warning:
            print("Converted multi-class probabilities to discrete labels.")

    return metrics.f1_score(y_true, y_pred, average=average)


def matthews_correlation(
    y_true: np.ndarray, y_pred: np.ndarray, warning=False
) -> float:
    """
    Calculate Matthews Correlation Coefficient (MCC).

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        float: Matthews Correlation Coefficient.
    """
    if (
        y_pred.ndim == 1
        and np.all((y_pred >= 0) & (y_pred <= 1))
        and not np.all(np.mod(y_pred, 1) == 0)
    ):
        y_pred = (y_pred >= 0.5).astype(int)
        if warning:
            print("Converted binary class probabilities to discrete labels.")
    elif (
        y_pred.ndim > 1
        and np.all((y_pred >= 0) & (y_pred <= 1))
        and not np.all(np.mod(y_pred, 1) == 0)
    ):
        y_pred = np.argmax(y_pred, axis=1)
        if warning:
            print("Converted multi-class probabilities to discrete labels.")

    return metrics.matthews_corrcoef(y_true, y_pred)


def top_k_accuracy(
    y_true: np.ndarray, y_pred: np.ndarray, k: Union[int, float]
) -> float:
    """
    Calculate top-k accuracy.

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted scores or probabilities for each class.
        k (int or float): Number or proportion of top predictions to consider.

    Returns:
        float: Top-k accuracy score.

    Raises:
        ValueError: If k is a float and not between 0 and 1.
    """
    if isinstance(k, float):
        if not (0 < k <= 1):
            raise ValueError("If k is a float, it must be between 0 and 1.")
        k = int(len(y_true) * k)
    k = min(k, len(y_true))
    top_k_preds = np.argsort(y_pred, axis=1)[:, -k:]
    matches = [y_true[i] in top_k_preds[i] for i in range(len(y_true))]
    return np.mean(matches)


def top_k_mae(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: Union[int, float],
    ascending: bool = False,
) -> float:
    """
    Calculate Mean Absolute Error (MAE) for the top k predictions.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        k (int or float): Number or proportion top predictions to consider.
        ascending (bool): If True, lower scores are better. If False, higher scores are better.

    Returns:
        float: MAE for the top k predictions.

    Raises:
        ValueError: If k is a float and not between 0 and 1.
    """
    if isinstance(k, float):
        if not (0 < k <= 1):
            raise ValueError("If k is a float, it must be between 0 and 1.")
        k = int(len(y_true) * k)
    k = min(k, len(y_true))
    if ascending:
        top_k_indices = np.argsort(y_pred)[:k]
    else:
        top_k_indices = np.argsort(y_pred)[-k:]
    return metrics.mean_absolute_error(y_true[top_k_indices], y_pred[top_k_indices])


def top_k_rmse(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: Union[int, float],
    ascending: bool = False,
) -> float:
    """
    Calculate Root Mean Squared Error (RMSE) for the top k predictions.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        k (int or float): Number or proportion of top predictions to consider.
        ascending (bool): If True, lower scores are better. If False, higher scores are better.

    Returns:
        float: MSE for the top k predictions.

    Raises:
        ValueError: If k is a float and not between 0 and 1.
    """
    if isinstance(k, float):
        if not (0 < k <= 1):
            raise ValueError("If k is a float, it must be between 0 and 1.")
        k = int(len(y_true) * k)
    k = min(k, len(y_true))
    if ascending:
        top_k_indices = np.argsort(y_pred)[:k]
    else:
        top_k_indices = np.argsort(y_pred)[-k:]
    return metrics.root_mean_squared_error(y_true[top_k_indices], y_pred[top_k_indices])


def top_k_recall(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: Union[int, float],
    ascending: bool = False,
) -> float:
    """
    Calculate the top-k recall for a regression model.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        k (int or float): Number or proportion of top predictions to consider.
        ascending (bool): If True, lower scores are better. If False, higher scores are better.

    Returns:
        float: Top-k recall.

    Raises:
        ValueError: If k is a float and not between 0 and 1.
    """
    if isinstance(k, float):
        if not (0 < k <= 1):
            raise ValueError("If k is a float, it must be between 0 and 1.")
        k = int(len(y_true) * k)
    k = min(k, len(y_true))

    if ascending:
        top_k_true_indices = np.argsort(y_true)[:k]
        top_k_pred_indices = np.argsort(y_pred)[:k]
    else:
        top_k_true_indices = np.argsort(y_true)[-k:]
        top_k_pred_indices = np.argsort(y_pred)[-k:]

    top_k_true_set = set(top_k_true_indices)
    top_k_pred_set = set(top_k_pred_indices)
    intersection = top_k_true_set.intersection(top_k_pred_set)

    recall = len(intersection) / k
    return recall


def selection_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    k: Union[int, float] = None,
    thresholds: list = None,
    ascending: bool = False,
) -> pd.DataFrame:
    """
    Generate a report with various metrics for a regression model evaluated as a compound selection task.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        k (int or float, optional): Number or proportion of top predictions to consider for top-k metrics.
        thresholds (list, optional): Threshold(s) to transform regression values into class labels.
            If a single int or float is provided, it is treated as a binary classification threshold.
            If a list or tuple is provided, it is sorted and used for multi-class classification.
        ascending (bool): If True, lower scores are better. If False, higher scores are better.

    Returns:
        pd.DataFrame: DataFrame containing the calculated metrics.

    Raises:
        ValueError: If neither k nor thresholds are provided.
    """
    if k is None and thresholds is None:
        raise ValueError("At least one of 'k' or 'thresholds' must be provided.")

    report = dict()

    if k is not None:
        report.update(
            {
                "spearman_correlation": spearman_correlation(y_true, y_pred),
                "top_k_mae": top_k_mae(y_true, y_pred, k=k, ascending=ascending),
                "top_k_rmse": top_k_rmse(y_true, y_pred, k=k, ascending=ascending),
                "top_k_recall": top_k_recall(y_true, y_pred, k=k, ascending=ascending),
                # "ndcg_at_k": ndcg_at_k(y_true, y_pred, k=k, ascending=ascending),
            }
        )

    if thresholds is not None:
        if isinstance(thresholds, (int, float)):
            thresholds = [thresholds]
        else:
            thresholds = sorted(thresholds)

        y_true_class = np.digitize(y_true, bins=thresholds)
        y_pred_class = np.digitize(y_pred, bins=thresholds)

        if len(thresholds) == 1:
            average = "binary"
        else:
            average = "weighted"

        report.update(
            {
                "accuracy": accuracy(y_true_class, y_pred_class),
                "precision": precision(y_true_class, y_pred_class, average=average),
                "recall": recall(y_true_class, y_pred_class, average=average),
                "f1_score": f1_score(y_true_class, y_pred_class, average=average),
                "MCC": matthews_correlation(y_true_class, y_pred_class),
            }
        )

    return report


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    average: str = "binary",
    multi_class: str = "raise",
) -> pd.DataFrame:
    """
    Generate a report with various metrics for a classification model.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.
        average (str): Type of averaging performed on the data.
        multi_class (str): Determines the type of averaging performed on the data for AUC-ROC and AUC-PRC.

    Returns:
        pd.DataFrame: DataFrame containing the calculated metrics.
    """
    report = {
        "accuracy": accuracy(y_true, y_pred),
        "precision": precision(y_true, y_pred, average=average),
        "recall": recall(y_true, y_pred, average=average),
        "f1_score": f1_score(y_true, y_pred, average=average),
        "MCC": matthews_correlation(y_true, y_pred),
    }

    if not np.all(np.mod(y_pred, 1) == 0):
        report["auc_roc"] = auc_roc(y_true, y_pred, multi_class=multi_class)
        report["auc_prc"] = auc_prc(y_true, y_pred, multi_class=multi_class)

    return report


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> pd.DataFrame:
    """
    Generate a report with various metrics for a regression model.

    Args:
        y_true (np.ndarray): True values.
        y_pred (np.ndarray): Predicted values.

    Returns:
        pd.DataFrame: DataFrame containing the calculated metrics.
    """
    report = {
        "r2_score": r2_score(y_true, y_pred),
        "root_mean_squared_error": root_mean_squared_error(y_true, y_pred),
        "mean_absolute_error": mean_absolute_error(y_true, y_pred),
        "pearson_correlation": pearson_correlation(y_true, y_pred),
        "spearman_correlation": spearman_correlation(y_true, y_pred),
    }
    return report
