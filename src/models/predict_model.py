import logging
import sys
import traceback

import torch
from transformers import HfArgumentParser, TrainingArguments, EarlyStoppingCallback
from transformers.integrations import MLflowCallback
from transformers.trainer_utils import denumpify_detensorize, IntervalStrategy, EvalPrediction

from constants import (
    COL_IS_ARGUMENT_PROB,
    COL_IS_ARGUMENT,
    COL_IS_STRONG_PROB,
)
from data_loader import ArgumentDataLoader
from metrics import (
    simple_regression_metric,
    simple_classification_metric_for_classes,
    average_metrics,
    optimal_f1_score_for_regression_on_binary_metric,
    optimal_f1_score_for_classification_on_continuous_metric,
)
from result_tracker import ResultTracker, DummyResultTracker
from task_arguments import (
    ModelInputArguments,
    PredictionTask,
    PredictionTaskArguments,
)
from trainer_factory import get_trainer_for_prediction_task, get_simple_regression_task_trainer


def get_label_col_for_task(task):
    # Same type for target and prediction
    if task == PredictionTask.STRENGTH_REGRESSION_EVALUATION:
        return COL_IS_STRONG_PROB
    if task == PredictionTask.ARGUMENT_PROB_REGRESSION_EVALUATION:
        return COL_IS_ARGUMENT_PROB
    if task == PredictionTask.ARGUMENT_DETECTION_EVALUATION:
        return COL_IS_ARGUMENT

    # Different type for target and prediction
    if (
        task == PredictionTask.STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION
        or task == PredictionTask.ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION
    ):
        return COL_IS_ARGUMENT_PROB
    if (
        task == PredictionTask.STRENGTH_REGRESSION_ARGUMENT_CORRELATION
        or task == PredictionTask.ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION
    ):
        return COL_IS_ARGUMENT
    if (
        task == PredictionTask.ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION
        or task == PredictionTask.ARGUMENT_DETECTION_STRENGTH_CORRELATION
    ):
        return COL_IS_STRONG_PROB

    # Transfer learning
    if task == PredictionTask.ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION:
        return COL_IS_ARGUMENT_PROB
    if task == PredictionTask.ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION:
        return COL_IS_STRONG_PROB

    raise ValueError(f"Unexpected task {task} when getting label column name!")


def get_split_dataset(task, model_name, include_motion, input_dataset_csv, max_len_percentile):
    label_col = get_label_col_for_task(task)

    loader = ArgumentDataLoader(model_name)
    return loader.create_and_format_split_dataset(
        input_dataset_csv,
        max_len_percentile,
        include_motion,
        label_col,
    )


def get_dataset(task, model_name, include_motion, input_dataset_csv, max_len_percentile):
    label_col = get_label_col_for_task(task)

    loader = ArgumentDataLoader(model_name)
    return loader.create_and_format_dataset(
        input_dataset_csv,
        max_len_percentile,
        include_motion,
        label_col,
    )


def do_prediction(trainer, dataset):
    test_loader = trainer.get_eval_dataloader(dataset)
    return trainer.evaluation_loop(test_loader, description="prediction", metric_key_prefix="test")


def simple_corr(task, batch_size, model_name, include_motion, input_dataset_csv, max_len_percentile, metric):
    trainer = get_trainer_for_prediction_task(task=task, batch_size=batch_size, model_name=model_name)
    dataset = get_dataset(task, model_name, include_motion, input_dataset_csv, max_len_percentile)
    all_splits_prediction = do_prediction(trainer, dataset)
    all_splits_metric = metric(all_splits_prediction)
    _, _, test_dataset = get_split_dataset(task, model_name, include_motion, input_dataset_csv, max_len_percentile)
    test_split_prediction = do_prediction(trainer, test_dataset)
    test_split_metric = metric(test_split_prediction)
    return {
        "all_splits_metric": all_splits_metric,
        "test_split_metric": test_split_metric,
    }


def mapped_regression_corr(task, batch_size, model_name, include_motion, input_dataset_csv, max_len_percentile, metric):
    trainer = get_trainer_for_prediction_task(task=task, batch_size=batch_size, model_name=model_name)
    train_dataset, eval_dataset, test_dataset = get_split_dataset(
        task, model_name, include_motion, input_dataset_csv, max_len_percentile
    )

    def add_output_as_inputs_col(dataset):
        prediction = do_prediction(trainer, dataset)
        prediction_column = [p.tolist() for p in prediction.predictions]
        new_dataset = dataset.add_column("inputs", prediction_column)
        new_dataset.set_format("torch", columns=["inputs", "labels"])
        return new_dataset

    train_dataset = add_output_as_inputs_col(train_dataset)
    eval_dataset = add_output_as_inputs_col(eval_dataset)
    test_dataset = add_output_as_inputs_col(test_dataset)

    def map_and_calculate_metric(map_fn, input_dim):
        if map_fn is None:
            mapped_train_dataset = train_dataset
            mapped_eval_dataset = eval_dataset
            mapped_test_dataset = test_dataset
        else:
            mapped_train_dataset = train_dataset.map(map_fn)
            mapped_eval_dataset = eval_dataset.map(map_fn)
            mapped_test_dataset = test_dataset.map(map_fn)
        train_args = TrainingArguments(
            output_dir=f"models/{str(task).lower()}-mapper",
            per_device_train_batch_size=64,
            per_device_eval_batch_size=64,
            save_total_limit=1,
            load_best_model_at_end=True,
            logging_strategy=IntervalStrategy.EPOCH,
            evaluation_strategy=IntervalStrategy.EPOCH,
            save_strategy=IntervalStrategy.EPOCH,
            num_train_epochs=100,
        )

        regression_trainer = get_simple_regression_task_trainer(
            train_args, mapped_train_dataset, mapped_eval_dataset, input_dim, 100, 1
        )
        early_stopping_callback = EarlyStoppingCallback(early_stopping_patience=5)
        regression_trainer.add_callback(early_stopping_callback)
        # Disable MLflow
        regression_trainer.remove_callback(MLflowCallback)
        regression_trainer.train()

        regression_prediction = do_prediction(regression_trainer, mapped_test_dataset)
        regression_metric = metric(regression_prediction)
        raw_metric = None
        if input_dim == 1:
            raw_metric = metric(
                EvalPrediction(predictions=mapped_test_dataset["inputs"], label_ids=mapped_test_dataset["labels"])
            )
        return regression_metric, raw_metric

    # Metrics
    regression_all_outputs, _ = map_and_calculate_metric(None, 2)
    regression_positive_output, raw_positive_output = map_and_calculate_metric(
        lambda x: {**x, "inputs": torch.tensor([x["inputs"][1]])}, 1
    )
    regression_negative_output, raw_negative_output = map_and_calculate_metric(
        lambda x: {**x, "inputs": torch.tensor([x["inputs"][0]])}, 1
    )
    regression_softmax_positive_output, raw_softmax_positive_output = map_and_calculate_metric(
        lambda x: {**x, "inputs": torch.tensor([torch.softmax(x["inputs"], dim=0)[1]])}, 1
    )
    regression_softmax_negative_output, raw_softmax_negative_output = map_and_calculate_metric(
        lambda x: {**x, "inputs": torch.tensor([torch.softmax(x["inputs"], dim=0)[0]])}, 1
    )

    return {
        "regression_metrics": {
            "all_outputs": regression_all_outputs,
            "positive_output": regression_positive_output,
            "negative_output": regression_negative_output,
            "softmax_positive_output": regression_softmax_positive_output,
            "softmax_negative_output": regression_softmax_negative_output,
        },
        "raw_metrics": {
            "positive_output": raw_positive_output,
            "negative_output": raw_negative_output,
            "softmax_positive_output": raw_softmax_positive_output,
            "softmax_negative_output": raw_softmax_negative_output,
        },
    }


def predict_single_model(task, batch_size, model_name, include_motion, input_dataset_csv, max_len_percentile, tracker):
    if (
        task == PredictionTask.ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION
        or task == PredictionTask.ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION
    ):
        return mapped_regression_corr(
            task=task,
            batch_size=batch_size,
            model_name=model_name,
            include_motion=include_motion,
            input_dataset_csv=input_dataset_csv,
            max_len_percentile=max_len_percentile,
            metric=simple_regression_metric,
        )

    if task == PredictionTask.ARGUMENT_DETECTION_EVALUATION:
        metric = simple_classification_metric_for_classes(["true", "false"])
    elif (
        task == PredictionTask.STRENGTH_REGRESSION_EVALUATION
        or task == PredictionTask.ARGUMENT_PROB_REGRESSION_EVALUATION
        or task == PredictionTask.STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION
        or task == PredictionTask.ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION
    ):
        metric = simple_regression_metric
    elif (
        task == PredictionTask.ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION
        or task == PredictionTask.STRENGTH_REGRESSION_ARGUMENT_CORRELATION
    ):
        metric = optimal_f1_score_for_regression_on_binary_metric
    elif (
        task == PredictionTask.ARGUMENT_DETECTION_STRENGTH_CORRELATION
        or task == PredictionTask.ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION
    ):
        metric = optimal_f1_score_for_classification_on_continuous_metric
    else:
        raise ValueError(f"Unexpected task {task} when getting metric!")

    return simple_corr(
        task=task,
        batch_size=batch_size,
        model_name=model_name,
        include_motion=include_motion,
        input_dataset_csv=input_dataset_csv,
        max_len_percentile=max_len_percentile,
        metric=metric,
    )


def predict(prediction_task_args, model_input_args, tracker):
    logging.info(f"Running task {prediction_task_args.task} ...")
    task = PredictionTask(prediction_task_args.task)
    batch_size = prediction_task_args.per_device_prediction_batch_size
    input_dataset_csv = model_input_args.input_dataset_csv
    max_len_percentile = model_input_args.max_len_percentile
    model_names = set(model_input_args.model_name)
    if "average" in model_names:
        raise ValueError('Unsupported model name "average", please choose something different.')

    results = {}
    for model_name in model_names:
        results[model_name] = predict_single_model(
            task=task,
            batch_size=batch_size,
            model_name=model_name,
            include_motion=model_input_args.include_motion,
            input_dataset_csv=input_dataset_csv,
            max_len_percentile=max_len_percentile,
            tracker=tracker,
        )

    if len(results) == 1:
        model_name = next(iter(model_names))
        results = results[model_name]
    else:
        results["average"] = average_metrics(list(results.values()))

    results = denumpify_detensorize(results)
    return results


def predict_and_track(prediction_task_args, model_input_args):
    if model_input_args.track_results:
        if model_input_args.result_file_name is None:
            raise ValueError("Please specify a result_file_name for prediction tasks.")
        tracker = ResultTracker(
            False,
            model_input_args.result_prefix,
            model_input_args.include_motion,
            prediction_task_args.task,
            model_input_args.result_file_name,
        )
    else:
        tracker = DummyResultTracker()
    tracker.write_command_line_section()
    try:
        result = predict(prediction_task_args, model_input_args, tracker)
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
    parser = HfArgumentParser([PredictionTaskArguments, ModelInputArguments])
    prediction_task_args, model_input_args = parser.parse_args_into_dataclasses()
    predict_and_track(prediction_task_args, model_input_args)
