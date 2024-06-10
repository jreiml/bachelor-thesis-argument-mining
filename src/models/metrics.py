from itertools import groupby

import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import f1_score, precision_score, recall_score, mean_squared_error
from transformers import EvalPrediction
from optimal_threshold_f1 import optimal_f1_score


def pearson(arr1, arr2):
    return np.corrcoef(arr1.flatten(), arr2.flatten())[0][1]


def spearman(arr1, arr2):
    return spearmanr(arr1, arr2).correlation


def classification_scores(golds, predictions, classes):
    f1_micro = f1_score(golds, predictions, average="micro")
    f1_macro = f1_score(golds, predictions, average="macro")
    f1_weighted = f1_score(golds, predictions, average="weighted")

    precision_micro = precision_score(golds, predictions, average="micro")
    precision_macro = precision_score(golds, predictions, average="macro")
    precision_weighted = precision_score(golds, predictions, average="weighted")

    recall_micro = recall_score(golds, predictions, average="micro")
    recall_macro = recall_score(golds, predictions, average="macro")
    recall_weighted = recall_score(golds, predictions, average="weighted")

    metrics = {
        "f1_micro": f1_micro,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
        "precision_micro": precision_micro,
        "precision_macro": precision_macro,
        "precision_weighted": precision_weighted,
        "recall_micro": recall_micro,
        "recall_macro": recall_macro,
        "recall_weighted": recall_weighted,
    }

    f1_score_class = f1_score(golds, predictions, average=None)
    for name, score in zip(classes, f1_score_class):
        metrics["f1_" + name] = score

    precision_score_class = precision_score(golds, predictions, average=None)
    for name, score in zip(classes, precision_score_class):
        metrics["precision_" + name] = score

    recall_score_class = recall_score(golds, predictions, average=None)
    for name, score in zip(classes, recall_score_class):
        metrics["recall_" + name] = score

    return metrics


def simple_regression_metric(eval_prediction):
    predictions = eval_prediction.predictions
    golds = eval_prediction.label_ids

    mse = mean_squared_error(golds, predictions, squared=True)
    rmse = mean_squared_error(golds, predictions, squared=False)
    spearman_value = spearman(golds, predictions)
    pearson_value = pearson(golds, predictions)

    return {
        "mean_squared_error": mse,
        "root_mean_squared_error": rmse,
        "spearman": spearman_value,
        "pearson": pearson_value,
    }


def simple_classification_metric_for_classes(classes):
    def simple_classification_metric(eval_prediction):
        predictions = np.argmax(eval_prediction.predictions, axis=1)
        golds = eval_prediction.label_ids
        return classification_scores(predictions, golds, classes)

    return simple_classification_metric


def optimal_f1_score_for_regression_on_binary_metric(eval_prediction: EvalPrediction):
    y_true = eval_prediction.label_ids
    y_score = eval_prediction.predictions
    threshold, f1_macro, f1_true, f1_false = optimal_f1_score(y_true=y_true, y_score=y_score)

    return {
        "threshold": threshold,
        "f1_macro": f1_macro,
        "f1_true": f1_true,
        "f1_false": f1_false,
    }


def optimal_f1_score_for_classification_on_continuous_metric(eval_prediction: EvalPrediction):
    y_true = np.argmax(eval_prediction.predictions, axis=1)
    y_score = eval_prediction.label_ids
    threshold, f1_macro, f1_true, f1_false = optimal_f1_score(y_true=y_true, y_score=y_score)

    return {
        "threshold": threshold,
        "f1_macro": f1_macro,
        "f1_true": f1_true,
        "f1_false": f1_false,
    }


def average_metrics(d):
    """
    Function to calculate the average of float-leaves in dicts. Modified to calculate mean/std

    Source: https://stackoverflow.com/questions/57311453/calculate-average-values-in-a-nested-dict-of-dicts
    """
    _data = sorted([i for b in d for i in b.items()], key=lambda x: x[0])
    _d = [(a, [j for _, j in b]) for a, b in groupby(_data, key=lambda x: x[0])]

    new_d = {}
    for a, b in _d:
        if isinstance(b[0], dict):
            new_d[a] = average_metrics(b)
        else:
            new_d[f"{a}_mean"] = np.mean(b)
            new_d[f"{a}_std"] = np.std(b)

    return new_d
