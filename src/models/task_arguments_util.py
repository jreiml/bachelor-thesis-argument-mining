from transformers.trainer_utils import IntervalStrategy

from task_arguments import TrainingTask


def convert_epoch_to_step_count(epoch, dataset_size, batch_size):
    """
    Helper method for converting an epoch to the amount of corresponding steps given the dataset and batch size.

    Params:
        epoch (float): The amount of epochs.
        dataset_size (int): The amount of samples in the dataset.
        batch_size (int): The amount of samples processed in a single batch.

    Returns:
        step_count (int): The amount of steps corresponding to the epoch, given the input parameters.
    """
    step_count = dataset_size / batch_size
    return int(step_count / (1 / epoch)) + 1


def set_epoch_step_for_training_args(train_args, train_dataset_len, logging_epoch_step):
    effective_batch_size = train_args.train_batch_size * train_args.gradient_accumulation_steps
    steps = convert_epoch_to_step_count(logging_epoch_step, train_dataset_len, effective_batch_size)
    train_args.logging_strategy = IntervalStrategy.STEPS
    train_args.evaluation_strategy = IntervalStrategy.STEPS
    train_args.save_strategy = IntervalStrategy.STEPS
    train_args.logging_steps = steps
    train_args.eval_steps = steps
    train_args.save_steps = steps
    train_args.warmup_steps = steps


def get_tasks_for_multitask(task):
    if task == TrainingTask.STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK:
        return [TrainingTask.STRENGTH_REGRESSION_TASK, TrainingTask.ARGUMENT_DETECTION_TASK]
    if task == TrainingTask.STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK:
        return [TrainingTask.STRENGTH_REGRESSION_TASK, TrainingTask.ARGUMENT_PROB_REGRESSION_TASK]
    if task == TrainingTask.ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK:
        return [TrainingTask.ARGUMENT_DETECTION_TASK, TrainingTask.ARGUMENT_PROB_REGRESSION_TASK]
    if task == TrainingTask.ALL_MULTI_TASK:
        return [
            TrainingTask.STRENGTH_REGRESSION_TASK,
            TrainingTask.ARGUMENT_DETECTION_TASK,
            TrainingTask.ARGUMENT_PROB_REGRESSION_TASK,
        ]

    raise ValueError(f"Unexpected task {task} when getting sub-tasks for multitask!")


def is_multitask(task):
    if task == TrainingTask.STRENGTH_REGRESSION_ARGUMENT_DETECTION_MULTI_TASK:
        return True
    if task == TrainingTask.STRENGTH_REGRESSION_ARGUMENT_PROB_REGRESSION_MULTI_TASK:
        return True
    if task == TrainingTask.ARGUMENT_DETECTION_ARGUMENT_PROB_REGRESSION_MULTI_TASK:
        return True
    if task == TrainingTask.ALL_MULTI_TASK:
        return True
    if task == TrainingTask.STRENGTH_REGRESSION_TASK:
        return False
    if task == TrainingTask.ARGUMENT_PROB_REGRESSION_TASK:
        return False
    if task == TrainingTask.ARGUMENT_DETECTION_TASK:
        return False

    raise ValueError(f"Unexpected task {task} when determining if is multi task!")
