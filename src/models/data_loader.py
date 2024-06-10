import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers.file_utils import PaddingStrategy
from transformers.tokenization_utils_base import TruncationStrategy

from constants import (
    COL_SET,
    COL_SENTENCE,
    COL_TOPIC,
    COL_IS_ARGUMENT,
    COL_IS_ARGUMENT_PROB,
    COL_IS_STRONG,
    COL_IS_STRONG_PROB,
    COL_STANCE,
    COL_STANCE_CONF,
    SET_TRAIN,
    SET_DEV,
    SET_TEST,
    STANCE_PRO_LABEL,
    STANCE_NO_ARG_LABEL,
    STANCE_CON_LABEL,
)


class ArgumentDataLoader:
    """
    The ArgumentDataLoader is a class, which simplifies the process of data preprocessing.
    It provides methods for creating datasets from csv files.
    """

    def __init__(self, pretrained_model_name_or_path):
        """
        Constructor. Initializes the tokenizer for a pretrained model.

        Parameters:
            pretrained_model_name_or_path (:obj:`str`):
                The model to use for the tokenizer.
        """
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, use_fast=False)

    def _format_dataset(self, dataset):
        dataset.set_format("torch", columns=["input_ids", "labels", "attention_mask"])

    def _split_dataset(self, dataset, splits):
        split = dataset.filter(lambda x: x[COL_SET] in splits)
        return split.flatten_indices()

    def _split_and_format_dataset(self, dataset, split):
        dataset_split = self._split_dataset(dataset, split)
        self._format_dataset(dataset_split)
        return dataset_split

    def _get_label(self, row, label_col):
        if label_col == COL_IS_ARGUMENT_PROB or label_col == COL_IS_STRONG_PROB or label_col == COL_STANCE_CONF:
            return row[label_col]

        if label_col == COL_IS_ARGUMENT or label_col == COL_IS_STRONG:
            return 1 if row[label_col] else 0

        if label_col != COL_STANCE:
            raise ValueError(f"Unexpected column {label_col} when getting label!")

        raw_label = row[label_col]
        if raw_label == STANCE_NO_ARG_LABEL:
            return 0
        if raw_label == STANCE_PRO_LABEL:
            return 1
        if raw_label == STANCE_CON_LABEL:
            return 2
        raise ValueError(f"Unexpected value {raw_label} for stance label!")

    def _get_label_dtype(self, label_col):
        if label_col == COL_IS_ARGUMENT:
            return torch.long
        if label_col == COL_IS_ARGUMENT_PROB:
            return torch.float
        if label_col == COL_IS_STRONG:
            return torch.long
        if label_col == COL_IS_STRONG_PROB:
            return torch.float
        if label_col == COL_STANCE:
            return torch.long
        if label_col == COL_STANCE_CONF:
            return torch.float

        raise ValueError(f"Unexpected column {label_col} when getting label dtype!")

    def _get_max_length(self, dataset, percentile, column):
        def get_length(batch):
            return self.tokenizer(batch, return_length=True, padding=False)

        lengths = dataset.map(get_length, input_columns=column, batched=True)["length"]
        return int(np.percentile(lengths, percentile)) + 1

    def _create_dataset(self, raw_dataset, percentile, include_motion, label_col=None):
        max_len = None if percentile == 100 else self._get_max_length(raw_dataset, percentile, COL_SENTENCE)

        def format_dataset_row(row):
            sentence = row[COL_SENTENCE]
            motion = row[COL_TOPIC] if include_motion else None
            encoding = self.tokenizer(
                sentence,
                motion,
                max_length=max_len,
                truncation=TruncationStrategy.ONLY_FIRST,
                padding=PaddingStrategy.LONGEST if max_len is None else PaddingStrategy.MAX_LENGTH,
                add_special_tokens=True,
                return_token_type_ids=False,
                return_attention_mask=True,
                return_tensors="pt",
            )
            formatted = {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
            }

            if label_col is not None:
                dtype = self._get_label_dtype(label_col)
                label = self._get_label(row, label_col)
                formatted["labels"] = torch.tensor([label], dtype=dtype)

            return formatted

        dataset = raw_dataset.map(format_dataset_row)

        return dataset

    def create_and_format_split_dataset(self, csv_files, percentile, include_motion, label_col=None):
        """
        Converts the given csv files into a train dataset, a dev dataset, and a test dataset.

        Parameters:
            csv_files (:obj:`str`):
                The path to the CSV files to be converted.
            percentile (:obj:`int`):
                The percentile for the maximum sentence length. Example:
                95 == The maximum sentence length is larger than the length 95% of all sentences.
            include_motion (:obj:`bool`):
                If the topic/motion should be included in the input.
            label_col (:obj:`str`, `optional`, defaults to None):
                The column name of the label within the dataset.

        Returns:
            train_dataset (:obj:`Dataset`):
                The dataset constructed from the train set.
            dev_dataset (:obj:`Dataset`):
                The dataset constructed from the dev set.
            test_dataset (:obj:`Dataset`):
                The dataset constructed from the test set.

        """
        raw_dataset = load_dataset("csv", data_files=csv_files)["train"]

        raw_train_dev_dataset = self._split_dataset(raw_dataset, {SET_TRAIN, SET_DEV})
        train_dev_dataset = self._create_dataset(raw_train_dev_dataset, percentile, include_motion, label_col)
        train_dataset = self._split_and_format_dataset(train_dev_dataset, {SET_TRAIN})
        dev_dataset = self._split_and_format_dataset(train_dev_dataset, {SET_DEV})

        raw_test_dataset = self._split_dataset(raw_dataset, {SET_TEST})
        test_dataset = self._create_dataset(raw_test_dataset, 100, include_motion, label_col)
        self._format_dataset(test_dataset)
        return train_dataset, dev_dataset, test_dataset

    def create_and_format_dataset(self, csv_files, percentile, include_motion, label_col=None):
        """
        Converts the given csv file into a dataset consisting of tensors.

        Parameters:
            csv_files (:obj:`str`):
                The path to the CSV files to be converted.
            percentile (:obj:`int`):
                The percentile for the maximum sentence length. Example:
                95 == The maximum sentence length is larger than the length 95% of all sentences.
            include_motion (:obj:`bool`):
                If the topic/motion should be included in the input.
            label_col (:obj:`str`, `optional`, defaults to None):
                The column name of the label within the dataset.
        Returns:
            dataset (:obj:`Dataset`):
                The dataset, which was extracted from the csv file.
        """
        raw_dataset = load_dataset("csv", data_files=csv_files)["train"]
        dataset = self._create_dataset(raw_dataset, percentile, include_motion, label_col)
        self._format_dataset(dataset)
        return dataset
