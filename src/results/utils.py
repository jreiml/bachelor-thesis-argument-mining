from itertools import groupby
import numpy as np


def average_results(d):
    """
    Function to calculate the average of float-leaves in dicts. Modified to calculate mean/std

    Source: https://stackoverflow.com/questions/57311453/calculate-average-values-in-a-nested-dict-of-dicts
    """
    _data = sorted([i for b in d for i in b.items()], key=lambda x: x[0])
    _d = [(a, [j for _, j in b]) for a, b in groupby(_data, key=lambda x: x[0])]

    new_d = {}
    for a, b in _d:
        if isinstance(b[0], dict):
            new_d[a] = average_results(b)
        else:
            new_d[f"{a}_mean"] = np.mean(b)
            new_d[f"{a}_std"] = np.std(b)

    return new_d
