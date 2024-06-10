import logging
import shutil
import sys
import traceback
from pathlib import Path

import transformers
from dotenv import load_dotenv
from transformers import (
    TrainingArguments,
    EarlyStoppingCallback,
)
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

from constants import COL_IS_STRONG_PROB, COL_IS_ARGUMENT_PROB, COL_IS_ARGUMENT
from data_loader import ArgumentDataLoader
from result_tracker import ResultTracker, DummyResultTracker
from task_arguments import (
    TrainingTaskArguments,
    TrainingTask,
    ModelInputArguments,
)
from task_arguments_util import is_multitask, get_tasks_for_multitask, set_epoch_step_for_training_args
from trainer_factory import get_trainer_for_training_task


def get_label_cols_for_multitask(multitask):
    return [get_label_col_for_task(task) for task in get_tasks_for_multitask(multitask)]


def get_label_col_for_task(task):
    if task == TrainingTask.STRENGTH_REGRESSION_TASK:
        return COL_IS_STRONG_PROB
    if task == TrainingTask.ARGUMENT_PROB_REGRESSION_TASK:
        return COL_IS_ARGUMENT_PROB
    if task == TrainingTask.ARGUMENT_DETECTION_TASK:
        return COL_IS_ARGUMENT

    raise ValueError(f"Unexpected task {task} when getting label column name!")


def delete_non_best_checkpoints(output_dir, best_model_checkpoint):
    glob_checkpoints = [str(x) for x in Path(output_dir).glob(f"{PREFIX_CHECKPOINT_DIR}-*")]
    for checkpoint in glob_checkpoints:
        if checkpoint != best_model_checkpoint:
            shutil.rmtree(checkpoint)


def get_multitask_datasets(task, model_input_args):
    model_name = model_input_args.model_name[0]
    label_cols = get_label_cols_for_multitask(task)
    tasks = get_tasks_for_multitask(task)

    input_dataset_csv = model_input_args.input_dataset_csv
    input_dataset_task_head = model_input_args.input_dataset_task_head

    label_task_head_indices = set(range(len(label_cols)))
    dataset_task_head_indices = set(input_dataset_task_head)

    if (
        label_task_head_indices != dataset_task_head_indices
        or len(input_dataset_csv) != len(input_dataset_task_head)
        or len(tasks) != len(label_cols)
    ):
        raise ValueError("Invalid task head configuration.")

    loader = ArgumentDataLoader(model_name)
    train_datasets, dev_datasets, test_datasets = {}, {}, {}
    for i, (task, label_col) in enumerate(zip(tasks, label_cols)):
        task_head_dataset_csv = [
            csv for task_head, csv in zip(input_dataset_task_head, input_dataset_csv) if task_head == i
        ]

        train_dataset, dev_dataset, test_dataset = loader.create_and_format_split_dataset(
            task_head_dataset_csv,
            model_input_args.max_len_percentile,
            model_input_args.include_motion,
            label_col,
        )
        task_name = str(task)
        train_datasets[task_name] = train_dataset
        dev_datasets[task_name] = dev_dataset
        test_datasets[task_name] = test_dataset

    return train_datasets, dev_datasets, test_datasets


def get_singletask_datasets(task, model_input_args):
    model_name = model_input_args.model_name[0]
    label_col = get_label_col_for_task(task)
    loader = ArgumentDataLoader(model_name)
    return loader.create_and_format_split_dataset(
        model_input_args.input_dataset_csv,
        model_input_args.max_len_percentile,
        model_input_args.include_motion,
        label_col,
    )


def get_task_head_loss_weights(training_task_args):
    task = TrainingTask(training_task_args.task)
    if not is_multitask(task):
        return None

    if training_task_args.task_head_loss_weights is None:
        raise ValueError("Please specify a task_head_loss_weights for multitask settings.")
    task_heads = get_tasks_for_multitask(task)
    if len(training_task_args.task_head_loss_weights) != len(task_heads):
        raise ValueError(f"Please specify exactly {len(task_heads)} task head loss weights.")
    task_head_loss_weights = dict()
    for i, task_head in enumerate(task_heads):
        task_head_loss_weights[task_head.name] = training_task_args.task_head_loss_weights[i]
    return task_head_loss_weights


def add_epoch_stats_to_results(results, state):
    current_epoch = state.epoch
    results["epochs_trained_total"] = current_epoch
    if state.best_model_checkpoint is None:
        return
    current_step = state.global_step
    steps_per_epoch = current_step / current_epoch
    checkpoint_number = int(state.best_model_checkpoint.split("-")[-1])
    checkpoint_epoch = checkpoint_number / steps_per_epoch
    results["epochs_trained_best"] = checkpoint_epoch


def train(training_task_args, train_args, model_input_args):
    if len(model_input_args.model_name) != 1:
        raise ValueError("For training tasks, exactly one model name has to be specified.")
    model_name = model_input_args.model_name[0]

    if training_task_args.mlflow_env is not None:
        load_dotenv(training_task_args.mlflow_env)

    task = TrainingTask(training_task_args.task)

    train_dataset, dev_dataset, test_dataset = (
        get_multitask_datasets(task, model_input_args)
        if is_multitask(task)
        else get_singletask_datasets(task, model_input_args)
    )

    logging_epoch_step = training_task_args.logging_epoch_step
    if logging_epoch_step is not None:
        train_dataset_len = sum([len(d) for d in train_dataset.values()]) if is_multitask(task) else len(train_dataset)
        set_epoch_step_for_training_args(train_args, train_dataset_len, logging_epoch_step)

    task_head_loss_weights = get_task_head_loss_weights(training_task_args)
    trainer = get_trainer_for_training_task(
        task=task,
        train_args=train_args,
        model_name=model_name,
        train_dataset=train_dataset,
        eval_dataset=dev_dataset,
        task_head_loss_weights=task_head_loss_weights,
    )
    early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=training_task_args.early_stopping_patience)
    trainer.add_callback(early_stopping_callback)
    trainer.train()

    if training_task_args.only_keep_best_model:
        delete_non_best_checkpoints(train_args.output_dir, trainer.state.best_model_checkpoint)

    # Log best eval results to MLFlow
    if training_task_args.mlflow_env is not None:
        trainer.evaluate(dev_dataset)

    # Log best test results to MLFlow and disk
    results = trainer.evaluate(test_dataset, metric_key_prefix="test")
    del results["test_runtime"], results["test_samples_per_second"], results["test_steps_per_second"], results["epoch"]
    add_epoch_stats_to_results(results, trainer.state)
    return results


def train_and_track(training_task_args, train_args, model_input_args):
    if model_input_args.track_results:
        tracker = ResultTracker(
            True,
            model_input_args.result_prefix,
            model_input_args.include_motion,
            training_task_args.task,
            train_args.run_name,
        )
    else:
        tracker = DummyResultTracker()
    tracker.write_command_line_section()
    try:
        result = train(training_task_args, train_args, model_input_args)
        logging.info(f"Result: {result}")
        tracker.write_result(result)
    except Exception:
        exc_type, exc_value, exc_traceback = sys.exc_info()
        lines = traceback.format_exception(exc_type, exc_value, exc_traceback)
        logging.error("".join(lines))
        tracker.write_exception(lines)
    tracker.finish()


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)
    parser = transformers.HfArgumentParser([TrainingTaskArguments, TrainingArguments, ModelInputArguments])
    training_task_args, train_args, model_input_args = parser.parse_args_into_dataclasses()
    train_and_track(training_task_args, train_args, model_input_args)
