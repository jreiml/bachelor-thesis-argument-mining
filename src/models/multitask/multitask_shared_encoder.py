"""
Alternative implementation of multitask in huggingface transformers using a shared encoder between models.
For this thesis we only required BertForSequenceClassification for all of our experiments, but this can be useful
if multiple different models need to be compared. Sample usage:

```
import os
from transformers import BertConfig, BertForSequenceClassification

def create_multitask_model(model_name_or_path, tasks):
    model_name_dict = {
        task.name: os.path.join(model_name_or_path, task.name)
        if os.path.exists(model_name_or_path)
        else model_name_or_path
        for task in tasks
    }

    model_type_dict = {task.name: BertForSequenceClassification for task in tasks}
    model_config_dict = {
        task.name: BertConfig.from_pretrained(model_name_dict[task.name], num_labels=task.num_labels) for task in tasks
    }
    return MultitaskModel.create(
        model_name_dict=model_name_dict, model_type_dict=model_type_dict, model_config_dict=model_config_dict
    )
```
"""

import os
from logging import getLogger
from typing import Optional

import torch
import torch.nn as nn
import transformers
from transformers import is_torch_tpu_available, PreTrainedModel
from transformers.file_utils import is_sagemaker_mp_enabled, WEIGHTS_NAME
from transformers.modeling_utils import unwrap_model
from transformers.trainer import TRAINING_ARGS_NAME
from transformers.trainer_utils import ShardedDDPOption

from .multitask_trainer import MultitaskTrainer

logger = getLogger(__name__)


class MultitaskModel(transformers.PreTrainedModel):
    def __init__(self, encoder, taskmodels_dict):
        """
        Setting MultitaskModel up as a PretrainedModel allows us
        to take better advantage of Trainer features
        """
        super().__init__(transformers.PretrainedConfig())

        self.encoder = encoder
        self.taskmodels_dict = nn.ModuleDict(taskmodels_dict)

    @classmethod
    def create(cls, model_name_dict, model_type_dict, model_config_dict):
        """
        This creates a MultitaskModel using the model class and config objects
        from single-task models.

        We do this by creating each single-task model, and having them share
        the same encoder transformer.
        """
        shared_encoder = None
        taskmodels_dict = {}
        for task_name, model_type in model_type_dict.items():
            model = model_type.from_pretrained(
                model_name_dict[task_name],
                config=model_config_dict[task_name],
            )
            if shared_encoder is None:
                shared_encoder = getattr(model, cls.get_encoder_attr_name(model))
            else:
                setattr(model, cls.get_encoder_attr_name(model), shared_encoder)
            taskmodels_dict[task_name] = model
        return cls(encoder=shared_encoder, taskmodels_dict=taskmodels_dict)

    @classmethod
    def get_encoder_attr_name(cls, model):
        """
        The encoder transformer is named differently in each model "architecture".
        This method lets us get the name of the encoder attribute
        """
        model_class_name = model.__class__.__name__
        if model_class_name.startswith("Bert"):
            return "bert"
        elif model_class_name.startswith("Roberta"):
            return "roberta"
        elif model_class_name.startswith("Albert"):
            return "albert"
        else:
            raise KeyError(f"Add support for new model {model_class_name}")

    def forward(self, task_name, **kwargs):
        return self.taskmodels_dict[task_name](**kwargs)


class MultitaskTrainerForSharedEncoderModel(MultitaskTrainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def save_model(self, output_dir: Optional[str] = None):
        """
        Will save the model, so you can reload it using `from_pretrained()`.
        Will only save from the main process.
        """

        if output_dir is None:
            output_dir = self.args.output_dir
        for task_name in self.task_configs:
            if is_torch_tpu_available():
                raise NotImplementedError("TPU saving is not implemented.")
            elif is_sagemaker_mp_enabled():
                raise NotImplementedError("Sagemaker MP saving is not implemented.")
            elif (
                ShardedDDPOption.ZERO_DP_2 in self.args.sharded_ddp
                or ShardedDDPOption.ZERO_DP_3 in self.args.sharded_ddp
            ):
                raise NotImplementedError("Sharded saving is not implemented.")
            elif self.deepspeed:
                raise NotImplementedError("Deepspeed saving is not implemented.")
            elif self.args.should_save:
                self._save(task_name, output_dir)

    def _save(self, task_name: str, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        task_output_dir = os.path.join(output_dir, task_name)

        os.makedirs(task_output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {task_output_dir}")
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model = self.model.taskmodels_dict[task_name]
        if not isinstance(model, PreTrainedModel):
            if isinstance(unwrap_model(model), PreTrainedModel):
                if state_dict is None:
                    state_dict = model.state_dict()
                unwrap_model(model).save_pretrained(task_output_dir, state_dict=state_dict)
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if state_dict is None:
                    state_dict = model.state_dict()
                torch.save(state_dict, os.path.join(task_output_dir, WEIGHTS_NAME))
        else:
            model.save_pretrained(task_output_dir, state_dict=state_dict)
        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(task_output_dir)
            # Also save to output dir, so we can also load the model checkpoint
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(task_output_dir, TRAINING_ARGS_NAME))
