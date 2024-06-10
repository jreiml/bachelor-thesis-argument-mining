"""
Implementation borrowed from transformers package and extended to support multiple prediction heads:

https://github.com/huggingface/transformers/blob/master/src/transformers/models/bert/modeling_bert.py
"""

import torch
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss
from transformers.file_utils import (
    add_code_sample_docstrings,
    add_start_docstrings_to_model_forward,
)
from transformers.modeling_outputs import SequenceClassifierOutput
from transformers.models.bert.modeling_bert import (
    BertPreTrainedModel,
    BERT_INPUTS_DOCSTRING,
    _TOKENIZER_FOR_DOC,
    _CHECKPOINT_FOR_DOC,
    _CONFIG_FOR_DOC,
    BertModel,
)


class MultitaskBertForSequenceClassification(BertPreTrainedModel):
    def __init__(self, config, **kwargs):
        super().__init__(config)
        self.task_configs = kwargs.get("task_configs", {})
        self.task_problem_type = {}
        self.config = config

        self.bert = BertModel(config)
        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)
        task_classifiers = {}
        for task in self.task_configs.values():
            task_classifiers[task.name] = task.init_layer_fn(config)
        self.task_classifiers = nn.ModuleDict(task_classifiers)

        num_tasks = len(self.task_configs)
        self.log_vars = nn.Parameter(torch.zeros(num_tasks))
        self.task_to_log_var_index = {task_name: i for i, task_name in enumerate(sorted(self.task_configs))}

        # Initialize weights and apply final processing
        self.post_init()

    # flake8: noqa: C901
    @add_start_docstrings_to_model_forward(BERT_INPUTS_DOCSTRING.format("batch_size, sequence_length"))
    @add_code_sample_docstrings(
        processor_class=_TOKENIZER_FOR_DOC,
        checkpoint=_CHECKPOINT_FOR_DOC,
        output_type=SequenceClassifierOutput,
        config_class=_CONFIG_FOR_DOC,
    )
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
        task_name=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size,)`, `optional`):
            Labels for computing the sequence classification/regression loss. Indices should be in :obj:`[0, ...,
            config.num_labels - 1]`. If :obj:`config.num_labels == 1` a regression loss is computed (Mean-Square loss),
            If :obj:`config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        logits = self.task_classifiers[task_name](pooled_output)
        loss = None
        if labels is not None:
            num_labels = self.task_configs[task_name].num_labels

            if task_name not in self.task_problem_type:
                if num_labels == 1:
                    self.task_problem_type[task_name] = "regression"
                elif num_labels > 1 and (labels.dtype == torch.long or labels.dtype == torch.int):
                    self.task_problem_type[task_name] = "single_label_classification"
                else:
                    self.task_problem_type[task_name] = "multi_label_classification"

            label_weights = self.task_configs[task_name].label_weights
            problem_type = self.task_problem_type[task_name]
            if problem_type == "regression":
                loss_fct = MSELoss()
                if num_labels == 1:
                    loss = loss_fct(logits.squeeze(), labels.squeeze())
                else:
                    loss = loss_fct(logits, labels)
            elif problem_type == "single_label_classification":
                loss_fct = CrossEntropyLoss(weight=label_weights)
                loss = loss_fct(logits.view(-1, num_labels), labels.view(-1))
            elif problem_type == "multi_label_classification":
                loss_fct = BCEWithLogitsLoss(weight=label_weights)
                loss = loss_fct(logits, labels)

        if loss is not None:
            loss = torch.mul(loss, self.task_configs[task_name].task_head_loss_weight)
            # Apply homoscedastic uncertainty to loss in training https://arxiv.org/pdf/1705.07115.pdf
            if self.training:
                index = self.task_to_log_var_index[task_name]
                precision = torch.exp(-self.log_vars[index])
                loss = precision * loss + self.log_vars[index]

        if not return_dict:
            output = (logits,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
