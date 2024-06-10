"""Determine optimal threshold for Macro F1 score."""
from typing import Tuple

import numpy as np
from sklearn.metrics._ranking import _binary_clf_curve


def f1_scores(
    precision: np.ndarray,
    recall: np.ndarray,
) -> np.ndarray:
    denom = precision + recall
    denom[denom == 0.0] = 1.0
    return 2 * (precision * recall) / denom


def recall(
    tps: np.ndarray,
    tps_fns: np.ndarray,
) -> np.ndarray:
    return tps / tps_fns


def precision(
    tps: np.ndarray,
    tps_fps: np.ndarray,
) -> np.ndarray:
    return tps / tps_fps


def all_f1_scores(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    # cf. https://stats.stackexchange.com/questions/518616/
    # cf. https://arxiv.org/abs/1911.03347
    # compute TP, FP, FN, TN for all thresholds
    fps, tps, thresholds = _binary_clf_curve(y_true, y_score)
    tns = fps[-1] - fps
    fns = tps[-1] - tps
    # F1-scores positive class
    f1_pos = f1_scores(
        precision=precision(tps=tps, tps_fps=tps + fps),
        recall=recall(tps=tps, tps_fns=tps[-1]),  # tps + fns = tps + (tps[-1] - tps) = tps[-1]
    )
    # F1-scores negative class
    f1_neg = f1_scores(
        precision=precision(tps=tns, tps_fps=tns + fns),
        recall=recall(tps=tns, tps_fns=tns + fps),  # tns + fps = fps[-1] - fps + fps = fps[-1]
    )
    # macro average
    f1 = 0.5 * (f1_pos + f1_neg)
    return thresholds, f1, f1_pos, f1_neg


def optimal_f1_score(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[float, float, float, float]:
    thresholds, f1s, f1s_pos, f1s_neg = all_f1_scores(y_true=y_true, y_score=y_score)
    idx = np.nanargmax(f1s)
    return thresholds[idx], f1s[idx], f1s_pos[idx], f1s_neg[idx]
