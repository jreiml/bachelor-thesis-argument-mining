import logging

from torch import nn
from transformers import BertForSequenceClassification, BertTokenizer, Trainer, TrainingArguments

from metrics import simple_regression_metric, simple_classification_metric_for_classes
from multitask.multitask_bert_model import MultitaskBertForSequenceClassification
from multitask.multitask_task_config import TaskConfig, metrics_for_tasks
from multitask.multitask_trainer import MultitaskPredictionTrainer, MultitaskTrainer
from regression_model import RegressionModel
from task_arguments import TrainingTask, PredictionTask
from task_arguments_util import get_tasks_for_multitask
from trainer_extension import PredictionTrainer


def get_simple_regression_task_trainer(train_args, train_dataset, eval_dataset, n_feature, n_hidden, n_output):
    def model_init():
        return RegressionModel(n_feature=n_feature, n_hidden=n_hidden, n_output=n_output)

    return Trainer(
        model_init=model_init,
        args=train_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=simple_regression_metric,
    )


def get_bert_regression_task_trainer(
    train_args, model_name, train_dataset=None, eval_dataset=None, for_prediction=False
):
    def model_init():
        return BertForSequenceClassification.from_pretrained(model_name, num_labels=1)

    tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=False)
    trainer_class = PredictionTrainer if for_prediction else Trainer
    metric = None if for_prediction else simple_regression_metric
    return trainer_class(
        model_init=model_init,
        args=train_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric,
    )


def get_bert_classification_task_trainer(
    train_args, model_name, classes, train_dataset=None, eval_dataset=None, for_prediction=False
):
    def model_init():
        return BertForSequenceClassification.from_pretrained(model_name, num_labels=len(classes))

    tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=False)
    trainer_class = PredictionTrainer if for_prediction else Trainer
    metric = None if for_prediction else simple_classification_metric_for_classes(classes)
    return trainer_class(
        model_init=model_init,
        args=train_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=metric,
    )


def get_multi_task_trainer(
    train_args,
    model_name,
    task_configs,
    train_dataset_dict=None,
    eval_dataset_dict=None,
    for_prediction=False,
):
    def model_init():
        return MultitaskBertForSequenceClassification.from_pretrained(model_name, task_configs=task_configs)

    tokenizer = BertTokenizer.from_pretrained(model_name, use_fast=False)
    trainer_class = MultitaskPredictionTrainer if for_prediction else MultitaskTrainer
    metric = None if for_prediction else metrics_for_tasks(task_configs.values())

    trainer = trainer_class(
        model_init=model_init,
        args=train_args,
        tokenizer=tokenizer,
        train_dataset=train_dataset_dict,
        eval_dataset=eval_dataset_dict,
        compute_metrics=metric,
        task_configs=task_configs,
    )
    return trainer


def get_task_config_for_task(task, task_head_loss_weights, train_dataset=None):
    task_name = str(task)
    if task == TrainingTask.ARGUMENT_DETECTION_TASK:
        metric = simple_classification_metric_for_classes(["true", "false"])
        num_labels = 2
    elif task == TrainingTask.STRENGTH_REGRESSION_TASK or task == TrainingTask.ARGUMENT_PROB_REGRESSION_TASK:
        metric = simple_regression_metric
        num_labels = 1
    else:
        raise ValueError(f"Unexpected task {task} when getting task config!")

    task_head_loss_weight = 1
    if task_name in task_head_loss_weights:
        task_head_loss_weight = task_head_loss_weights[task_name]
    else:
        logging.warning(f"Task {task_name} has no loss weight specified, using default weight of 1.")

    label_weights = None
    # Unused, but could be enabled for certain tasks
    # if num_labels > 1 and train_dataset is not None:
    #    label_weights = compute_label_weights(train_dataset)

    def init_layer_fn(config):
        return nn.Linear(config.hidden_size, num_labels)

    return TaskConfig(task_name, metric, num_labels, task_head_loss_weight, label_weights, init_layer_fn)


def get_trainer_for_training_task(
    task, train_args, model_name, train_dataset=None, eval_dataset=None, task_head_loss_weights=None
):
    if task == TrainingTask.STRENGTH_REGRESSION_TASK or task == TrainingTask.ARGUMENT_PROB_REGRESSION_TASK:
        if task_head_loss_weights is not None:
            raise ValueError("task_head_loss_weights is unsupported for single-task settings.")
        return get_bert_regression_task_trainer(
            train_args=train_args,
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
        )

    if task == TrainingTask.ARGUMENT_DETECTION_TASK:
        if task_head_loss_weights is not None:
            raise ValueError("task_head_loss_weights is unsupported for single-task settings.")
        return get_bert_classification_task_trainer(
            train_args=train_args,
            model_name=model_name,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            classes=["true", "false"],
        )

    if (
        task == TrainingTask.STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK
        or task == TrainingTask.STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK
        or task == TrainingTask.ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK
        or task == TrainingTask.ALL_MULTI_TASK
    ):
        tasks = get_tasks_for_multitask(task)
        task_configs = {
            task.name: get_task_config_for_task(task, task_head_loss_weights, train_dataset[task.name])
            for task in tasks
        }
        return get_multi_task_trainer(
            train_args=train_args,
            model_name=model_name,
            task_configs=task_configs,
            train_dataset_dict=train_dataset,
            eval_dataset_dict=eval_dataset,
        )

    raise ValueError(f"Unexpected task {task} when getting Trainer!")


def get_trainer_for_prediction_task(task, batch_size, model_name):
    train_args = TrainingArguments(per_device_eval_batch_size=batch_size, output_dir="tmp_trainer")
    if (
        task == PredictionTask.STRENGTH_REGRESSION_EVALUATION
        or task == PredictionTask.ARGUMENT_PROB_REGRESSION_EVALUATION
        or task == PredictionTask.STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION
        or task == PredictionTask.STRENGTH_REGRESSION_ARGUMENT_CORRELATION
        or task == PredictionTask.ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION
        or task == PredictionTask.ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION
    ):
        return get_bert_regression_task_trainer(
            train_args=train_args,
            model_name=model_name,
            for_prediction=True,
        )

    if (
        task == PredictionTask.ARGUMENT_DETECTION_EVALUATION
        or task == PredictionTask.ARGUMENT_DETECTION_STRENGTH_CORRELATION
        or task == PredictionTask.ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION
        or task == PredictionTask.ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION
        or task == PredictionTask.ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION
    ):
        return get_bert_classification_task_trainer(
            train_args=train_args,
            model_name=model_name,
            classes=["true", "false"],
            for_prediction=True,
        )

    raise ValueError(f"Unexpected task {task} when getting trainer for prediction task!")
