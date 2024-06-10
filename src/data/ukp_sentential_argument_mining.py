import logging
import os
from _csv import QUOTE_NONE
from os import listdir
import random

import pandas as pd

from constants import (
    COL_IS_ARGUMENT,
    STANCE_PRO_LABEL,
    STANCE_CON_LABEL,
    COL_STANCE,
    COL_ID,
    STANCE_NO_ARG_LABEL,
    SET_DEV,
    SET_TRAIN,
    SET_TEST,
    COL_TOPIC,
    COL_SET,
)
from util import adjust_dataset_format, invert_stance, in_and_cross_topic_to_output

DATASET_INPUT_DIRECTORY = "UKP_sentential_argument_mining/data"
IN_TOPIC_DATASET_OUTPUT_FILE = "ukp_sentential_argument_mining.csv"
CROSS_TOPIC_DATASET_OUTPUT_FILE = "ukp_sentential_argument_mining_cross_topic.csv"
ALL_CROSS_TOPIC_OUTPUT_DIRECTORY = "ukp_cross_topic"
CROSS_TOPIC_SHUFFLE_SEED = 3978468931496

RAW_COL_ANNOTATION = "annotation"
RAW_SET_VAL = "val"
RAW_ANNOTATION_NO_ARG = "NoArgument"
RAW_ANNOTATION_AGAINST = "Argument_against"
RAW_ANNOTATION_FOR = "Argument_for"

FILE_TO_TOPIC = {
    # Evidences sentences
    "abortion.tsv": "We should ban abortions",
    # Arg Rank 30k, Evidences sentences
    "cloning.tsv": "We should ban human cloning",
    # Arg Rank 30k, Evidences sentences
    "death_penalty.tsv": "We should abolish capital punishment",
    # Evidences sentences
    "gun_control.tsv": "We should increase gun control",
    # Arg Rank 30k, Evidences sentences
    "marijuana_legalization.tsv": "We should legalize cannabis",
    # Custom
    "minimum_wage.tsv": "We should introduce a minimum wage",
    # Evidences sentences
    "nuclear_energy.tsv": "We should further exploit nuclear power",
    # Evidences sentences
    "school_uniforms.tsv": "We should ban school uniforms",
}

TOPIC_TO_FILE_NAME = {
    "We should ban abortions": "abortion",
    "We should ban human cloning": "cloning",
    "We should abolish capital punishment": "death_penalty",
    "We should increase gun control": "gun_control",
    "We should legalize cannabis": "marijuana_legalization",
    "We should introduce a minimum wage": "minimum_wage",
    "We should further exploit nuclear power": "nuclear_energy",
    "We should ban school uniforms": "school_uniforms",
}

FILE_TO_IS_INVERTED_STANCE = {
    "abortion.tsv": True,
    "cloning.tsv": True,
    "death_penalty.tsv": True,
    "gun_control.tsv": False,
    "marijuana_legalization.tsv": False,
    "minimum_wage.tsv": False,
    "nuclear_energy.tsv": False,
    "school_uniforms.tsv": True,
}

TOPIC_TO_SPLIT = {
    "We should ban abortions": SET_TRAIN,
    "We should ban human cloning": SET_DEV,
    "We should abolish capital punishment": SET_TRAIN,
    "We should increase gun control": SET_TRAIN,
    "We should legalize cannabis": SET_TEST,
    "We should introduce a minimum wage": SET_TEST,
    "We should further exploit nuclear power": SET_TRAIN,
    "We should ban school uniforms": SET_TRAIN,
}

logger = logging.getLogger(__file__)


def map_set(set_split):
    if set_split == RAW_SET_VAL:
        return SET_DEV

    if set_split == SET_TRAIN or set_split == SET_TEST:
        return set_split

    raise ValueError(f"Unexpected value {set_split} when mapping set!")


def map_annotation_to_stance(annotation):
    if annotation == RAW_ANNOTATION_NO_ARG:
        return STANCE_NO_ARG_LABEL

    if annotation == RAW_ANNOTATION_AGAINST:
        return STANCE_CON_LABEL

    if annotation == RAW_ANNOTATION_FOR:
        return STANCE_PRO_LABEL

    raise ValueError(f"Unexpected value {annotation} when mapping annotation!")


def map_annotation_to_is_argument(annotation):
    if annotation == RAW_ANNOTATION_NO_ARG:
        return False

    if annotation == RAW_ANNOTATION_AGAINST or annotation == RAW_ANNOTATION_FOR:
        return True

    raise ValueError(f"Unexpected value {annotation} when mapping annotation!")


def map_annotation_to_stance_inverted(annotation):
    return invert_stance(map_annotation_to_stance(annotation))


def get_stance_fn_for_file(file):
    inverted = FILE_TO_IS_INVERTED_STANCE[file]
    if inverted:
        return map_annotation_to_stance_inverted
    return map_annotation_to_stance


def load_dataset(input_filepath):
    input_directory = os.path.join(input_filepath, DATASET_INPUT_DIRECTORY)
    raw_dfs = []
    for file in sorted(listdir(input_directory)):
        raw_dataset_path = os.path.join(input_directory, file)
        raw_df = pd.read_csv(raw_dataset_path, delimiter="\t", quoting=QUOTE_NONE)
        raw_dfs.append((raw_df, file))

    return raw_dfs


def process_single_topic(raw_df, file):
    stance_fn = get_stance_fn_for_file(file)

    raw_df[COL_STANCE] = raw_df[RAW_COL_ANNOTATION].map(stance_fn)
    raw_df[COL_IS_ARGUMENT] = raw_df[RAW_COL_ANNOTATION].map(map_annotation_to_is_argument)
    raw_df[COL_SET] = raw_df[COL_SET].map(map_set)
    raw_df[COL_TOPIC] = FILE_TO_TOPIC[file]
    return raw_df


def process(input_filepath, output_filepath, topic_split_restriction):
    for topic, split in TOPIC_TO_SPLIT.items():
        if topic in topic_split_restriction and topic_split_restriction[topic] != split:
            raise AssertionError("Unexpected topic split restriction for UKP!")

    logger.info("Loading dataset ...")
    raw_dfs = load_dataset(input_filepath)
    logger.info("... done!")

    logger.info("Processing ...")
    processed_dfs = [process_single_topic(raw_df, file) for (raw_df, file) in raw_dfs]
    combined_df = pd.concat(processed_dfs, ignore_index=True)
    combined_df = adjust_dataset_format(combined_df)
    combined_df.index.name = COL_ID
    new_restriction = in_and_cross_topic_to_output(
        df=combined_df,
        seed=None,
        topic_split_restriction=TOPIC_TO_SPLIT,
        output_filepath=output_filepath,
        in_topic_output=IN_TOPIC_DATASET_OUTPUT_FILE,
        cross_topic_output=CROSS_TOPIC_DATASET_OUTPUT_FILE,
    )

    cross_topic_dir = os.path.join(output_filepath, ALL_CROSS_TOPIC_OUTPUT_DIRECTORY)
    if not os.path.exists(cross_topic_dir):
        os.makedirs(cross_topic_dir)

    topics_for_test = list(set(combined_df[COL_TOPIC]))
    topics_for_dev = list(topics_for_test)
    random.seed(CROSS_TOPIC_SHUFFLE_SEED)

    for topic_for_dev in topics_for_dev:
        for topic_for_test in topics_for_test:
            if topic_for_dev == topic_for_test:
                continue
            combined_df[COL_SET] = SET_TRAIN
            combined_df.loc[combined_df[COL_TOPIC] == topic_for_dev, COL_SET] = SET_DEV
            combined_df.loc[combined_df[COL_TOPIC] == topic_for_test, COL_SET] = SET_TEST
            file_name = f"{TOPIC_TO_FILE_NAME[topic_for_dev]}_dev_{TOPIC_TO_FILE_NAME[topic_for_test]}_test.csv"
            output = os.path.join(output_filepath, ALL_CROSS_TOPIC_OUTPUT_DIRECTORY, file_name)
            combined_df.to_csv(output)

    logger.info("... done!")
    return new_restriction
