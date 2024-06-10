from collections import defaultdict

import numpy as np
from transformers import EvalPrediction


class TaskConfig:
    def __init__(self, name, metric, num_labels, task_head_loss_weight, label_weights, init_layer_fn):
        self.name = name
        self.metric = metric
        self.num_labels = num_labels
        self.task_head_loss_weight = task_head_loss_weight
        self.label_weights = label_weights
        self.init_layer_fn = init_layer_fn


def metrics_for_tasks(task_configs):
    def metric(eval_prediction, task_names):
        task_to_indices = defaultdict(list)
        for i, task_name in enumerate(task_names):
            task_to_indices[task_name].append(i)

        metrics = {}
        for task in task_configs:
            indices = task_to_indices[task.name]
            task_label_ids = np.take(eval_prediction.label_ids, indices, axis=0)
            task_predictions = np.take(eval_prediction.predictions, indices, axis=0)
            if np.size(task_predictions, 1) > task.num_labels:
                task_predictions = task_predictions[:, : task.num_labels]
            task_eval_prediction = EvalPrediction(predictions=task_predictions, label_ids=task_label_ids)
            for key, value in task.metric(task_eval_prediction).items():
                metrics[f"{task.name}-{key}"] = value
        return metrics

    return metric
