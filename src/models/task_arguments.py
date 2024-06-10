from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class TrainingTask(Enum):
    STRENGTH_REGRESSION_TASK = "STRENGTH_REGRESSION_TASK"
    ARGUMENT_PROB_REGRESSION_TASK = "ARGUMENT_PROB_REGRESSION_TASK"
    ARGUMENT_DETECTION_TASK = "ARGUMENT_DETECTION_TASK"
    STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK = "STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK"
    STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK = "STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK"
    ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK = "ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK"
    ALL_MULTI_TASK = "ALL_MULTI_TASK"

    def __str__(self):
        return self.value


class PredictionTask(Enum):
    # Same type for target and prediction
    STRENGTH_REGRESSION_EVALUATION = "STRENGTH_REGRESSION_EVALUATION"
    ARGUMENT_PROB_REGRESSION_EVALUATION = "ARGUMENT_PROB_REGRESSION_EVALUATION"
    ARGUMENT_DETECTION_EVALUATION = "ARGUMENT_DETECTION_EVALUATION"

    # Different type for target and prediction
    STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION = "STRENGTH_REGRESSION_ARGUMENT_PROB_CORRELATION"
    STRENGTH_REGRESSION_ARGUMENT_CORRELATION = "STRENGTH_REGRESSION_ARGUMENT_CORRELATION"
    ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION = "ARGUMENT_PROB_REGRESSION_STRENGTH_CORRELATION"
    ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION = "ARGUMENT_PROB_REGRESSION_ARGUMENT_CORRELATION"
    ARGUMENT_DETECTION_STRENGTH_CORRELATION = "ARGUMENT_DETECTION_STRENGTH_CORRELATION"
    ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION = "ARGUMENT_DETECTION_ARGUMENT_PROB_CORRELATION"

    # Transfer learning
    ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION = "ARGUMENT_DETECTION_TRANSFER_ARG_PROB_CORRELATION"
    ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION = "ARGUMENT_DETECTION_TRANSFER_STRENGTH_CORRELATION"

    def __str__(self):
        return self.value


@dataclass
class TrainingTaskArguments:
    """
    TrainingTaskArguments is the subset of the arguments we use in our scripts
    **which relate to the training of certain tasks**.

    Using :class:`~transformers.HfArgumentParser` we can turn this class into `argparse
    <https://docs.python.org/3/library/argparse.html#module-argparse>`__ arguments that
    can be specified on the command line.

    Parameters:
        task (:obj:`TrainingTask`):
            The pre-defined task, which the model should be trained on.
        mlflow_env (:obj:`str`, `optional`, defaults to None):
            An optional env file containing the necessary environment
            variables for enabling Mlflow.
        early_stopping_patience (:obj:`int`, `optional`, defaults to 5):
            How often the eval loss can get worse before stopping training.
        only_keep_best_model (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If all checkpoints except for the best one should be deleted after training.
        logging_epoch_step (:obj:`float`, `optional`, defaults to None):
            In what epoch-interval the model should evaluate the model and log the results.
            Additionally, it will also use the first epoch step as a warm-up period.
            If specified, overwrites warmup_steps, eval_steps, logging_steps and save_steps for the TrainingArguments.
    """

    task: TrainingTask = field(
        metadata={"help": "The pre-defined task, which the model should be trained on."},
    )
    task_head_loss_weights: List[float] = field(
        default=None,
        metadata={
            "help": "The multiplier for the loss of each task head in a multi-task setting. "
            "We use a weighted sum for the combined loss."
        },
    )
    mlflow_env: Optional[str] = field(
        default=None,
        metadata={
            "help": ("An optional env file containing the necessary environment" "variables for enabling Mlflow.")
        },
    )
    early_stopping_patience: int = field(
        default=25,
        metadata={"help": "How often the eval loss can get worse before stopping training."},
    )
    only_keep_best_model: bool = field(
        default=True,
        metadata={"help": "If all checkpoints except for the best one should be deleted after training."},
    )
    logging_epoch_step: Optional[float] = field(
        default=None,
        metadata={
            "help": "In what epoch-interval the model should evaluate the model and log the results."
            "Additionally, it will also use the first epoch step as a warm-up period."
            "If specified, overwrites warmup_steps, eval_steps, logging_steps and save_steps for the TrainingArguments."
        },
    )


@dataclass
class PredictionTaskArguments:
    """
    PredictionTaskArguments is the subset of the arguments we use in our scripts
    **which relate to the prediction for certain tasks**.

    Using :class:`~transformers.HfArgumentParser` we can turn this class into `argparse
    <https://docs.python.org/3/library/argparse.html#module-argparse>`__ arguments that
    can be specified on the command line.

    Parameters:
        task (:obj:`PredictionTask`):
            The pre-defined task, which the model should do predictions on.
        per_device_prediction_batch_size (:obj:`int`, `optional`, defaults to 8):
            The batch size per GPU/TPU core/CPU for prediction.
    """

    task: PredictionTask = field(
        metadata={"help": "The pre-defined task, which the model should do predictions on."},
    )
    per_device_prediction_batch_size: int = field(
        default=8,
        metadata={"help": "The batch size per GPU/TPU core/CPU for prediction."},
    )


@dataclass
class ModelInputArguments:
    """
    ModelInputArguments is the subset of the arguments we use in our scripts
    **which relate to the model and its input**.

    Using :class:`~transformers.HfArgumentParser` we can turn this class into `argparse
    <https://docs.python.org/3/library/argparse.html#module-argparse>`__ arguments that
    can be specified on the command line.

    Parameters:
        input_dataset_csv (:obj:`List[str]`):
            The list of files, which are the datasets to be trained on.
        model_name (:obj:`List[str]`):
            The pretrained model/tokenizer to use.
            Multiple model names can only be specified for prediction tasks, the average metrics will be returned!
        input_dataset_task_head (:obj:`List[int]`, `optional`, defaults to None):
            The list of indices indicating which dataset belong to which task head in a multi-task setting.
            Needs to match the size of `input_dataset_csv`, if specified.
        max_len_percentile (:obj:`int`, `optional`, defaults to 100):
            The highest percentile of sentence length which is considered.
            Longer sentences are truncated.
        include_motion (:obj:`bool`, `optional`, defaults to :obj:`False`):
            If the motion/topic is included in the model input during training.
        track_results (:obj:`bool`, `optional`, defaults to :obj:`True`):
            If the results should be tracked on the disk in a markdown file.
        result_prefix (:obj:`str`, `optional`, defaults to None):
            If results are tracked, where they are stored in the `results` folder.
            If not specified, they are written relative to the `results` folder.
        result_file_name (:obj:`str`, `optional`, defaults to None):
            If results are tracked, the name of the result output file.
            If not specified for training, the run name is used.
            Must be specified for prediction tasks.
    """

    input_dataset_csv: List[str] = field(metadata={"help": "The list of files, which are the datasets to be used."})
    model_name: List[str] = field(
        metadata={
            "help": "The pretrained model/tokenizer to use. Multiple model names can only be specified for "
            "prediction tasks, the average metrics will be returned!"
        },
    )
    input_dataset_task_head: List[int] = field(
        default=None,
        metadata={
            "help": "The list of indices indicating which dataset belong to which task head in a multi-task setting."
            " Needs to match the size of `input_dataset_csv`, if specified."
        },
    )
    max_len_percentile: int = field(
        default=100,
        metadata={
            "help": "The highest percentile of sentence length which is considered." "Longer sentences are truncated."
        },
    )
    include_motion: bool = field(
        default=False,
        metadata={"help": "If the motion/topic is included in the model input during training."},
    )
    track_results: bool = field(
        default=True,
        metadata={"help": "If the results should be tracked on the disk in a markdown file."},
    )
    result_prefix: Optional[str] = field(
        default=None,
        metadata={
            "help": "If results are tracked, where they are stored in the `results` folder."
            "If not specified, they are written relative to the `results` folder."
        },
    )
    result_file_name: Optional[str] = field(
        default=None,
        metadata={
            "help": "If results are tracked, the name of the result output file. "
            "If not specified for training, the run name is used."
            "Must be specified for prediction tasks."
        },
    )
