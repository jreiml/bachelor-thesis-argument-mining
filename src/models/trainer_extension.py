from collections import defaultdict

import torch
from transformers import Trainer


def compute_label_weights(dataset, to_device=True):
    """
    Helper method for calculating relative weights for each label.
    Can be used to improving the loss function in training,
    so it takes into account if a class has less instances than another (and therefore has a weigher weight).

    Params:
        dataset (Dataset): The dataset which contains all training instances.
        to_device (bool): Call `.to(device)` on weight tensor, to prepare it for cuda, if available.

    Returns:
        label_weights (List[int]): A list of length num_labels, with each entry representing the weight of a class.
    """

    label_counts = defaultdict(int)
    for label in dataset["labels"]:
        label_counts[label.item()] += 1
    num_labels = max(label_counts) + 1
    total_count = sum(label_counts.values())
    weights = [
        (1 / num_labels) / (label_counts[label] / total_count) if label in label_counts else 0
        for label in range(num_labels)
    ]
    weights_tensor = torch.tensor(weights, dtype=torch.float)
    if to_device:
        device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
        weights_tensor = weights_tensor.to(device)
    return weights_tensor


class PredictionTrainer(Trainer):
    """
    The PredictionTrainer overwrites the compute_loss function, and skips the loss calculation

    We do not care about the loss for predictions.
    """

    def compute_loss(self, model, inputs, return_outputs=False):
        loss = torch.empty(0)
        if not return_outputs:
            return loss
        # Do not pass labels to model, otherwise it calculates the loss for the model
        if "labels" in inputs:
            del inputs["labels"]
        outputs = model(**inputs)
        return loss, outputs


class MultilabelTrainer(Trainer):
    """
    The MultilabelTrainer implements a custom loss function,
    which allows the weighting of labels when calculating the total loss.

    This can be useful if the classes in our dataset are heavily skewed in one direction.
    """

    def __init__(self, train_dataset, calculate_label_weights=True, *args, **kwargs):
        super().__init__(train_dataset=train_dataset, *args, **kwargs)
        label_weights = None

        if calculate_label_weights:
            if train_dataset is None:
                raise ValueError("Train Dataset has to be passed for MultilabelTrainer!")
            label_weights = compute_label_weights(train_dataset)

        self.loss_fct = torch.nn.CrossEntropyLoss(weight=label_weights)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss
