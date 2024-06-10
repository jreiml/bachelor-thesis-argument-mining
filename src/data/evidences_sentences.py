import logging
import os

import pandas as pd

from constants import (
    COL_SENTENCE,
    COL_IS_ARGUMENT_PROB,
    COL_ID,
    COL_TOPIC,
)
from util import add_set_column, adjust_dataset_format, in_and_cross_topic_to_output

DATASET_INPUT_FILE = "evidences_sentences/wikipedia_evidence_dataset_29429.csv"
IN_TOPIC_DATASET_OUTPUT_FILE = "evidences_sentences.csv"
CROSS_TOPIC_DATASET_OUTPUT_FILE = "evidences_sentences_cross_topic.csv"

IN_TOPIC_SET_DISTRIBUTION_SEED = 294297382516
CROSS_TOPIC_SET_DISTRIBUTION_SEED = 349086734806

RAW_COL_MOTION_TEXT = "Motion Text"
RAW_COL_SENTENCE = "Evidence"
RAW_COL_ACCEPTANCE_RATE = "acceptanceRate"

logger = logging.getLogger(__file__)


def load_dataset(input_filepath):
    raw_dataset_path = os.path.join(input_filepath, DATASET_INPUT_FILE)
    raw_df = pd.read_csv(raw_dataset_path)
    return raw_df


def process(input_filepath, output_filepath, topic_split_restriction=None):
    logger.info("Loading dataset ...")
    raw_df = load_dataset(input_filepath)
    logger.info("... done!")

    logger.info("Processing ...")
    raw_df.index.name = COL_ID
    add_set_column(IN_TOPIC_SET_DISTRIBUTION_SEED, raw_df)

    raw_df = raw_df.rename(
        columns={
            RAW_COL_MOTION_TEXT: COL_TOPIC,
            RAW_COL_SENTENCE: COL_SENTENCE,
            RAW_COL_ACCEPTANCE_RATE: COL_IS_ARGUMENT_PROB,
        }
    )
    raw_df = adjust_dataset_format(raw_df)
    new_restriction = in_and_cross_topic_to_output(
        df=raw_df,
        seed=CROSS_TOPIC_SET_DISTRIBUTION_SEED,
        topic_split_restriction=topic_split_restriction,
        output_filepath=output_filepath,
        in_topic_output=IN_TOPIC_DATASET_OUTPUT_FILE,
        cross_topic_output=CROSS_TOPIC_DATASET_OUTPUT_FILE,
    )
    logger.info("... done!")
    return new_restriction
