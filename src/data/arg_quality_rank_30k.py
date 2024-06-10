import logging
import os

import pandas as pd

from constants import (
    COL_IS_ARGUMENT,
    STANCE_PRO_LABEL,
    STANCE_CON_LABEL,
    COL_STANCE,
    COL_STANCE_CONF,
    COL_IS_STRONG_PROB,
    COL_ID,
    COL_SENTENCE,
    COL_TOPIC,
    COL_SET,
)
from util import adjust_dataset_format, in_and_cross_topic_to_output, add_set_column

DATASET_INPUT_FILE = "arg_quality_rank_30k/arg_quality_rank_30k.csv"
IN_TOPIC_DATASET_WA_OUTPUT_FILE = "arg_quality_rank_30k_wa.csv"
IN_TOPIC_DATASET_MACE_P_OUTPUT_FILE = "arg_quality_rank_30k_mace_p.csv"
CROSS_TOPIC_DATASET_WA_OUTPUT_FILE = "arg_quality_rank_30k_cross_topic_wa.csv"
CROSS_TOPIC_DATASET_MACE_P_OUTPUT_FILE = "arg_quality_rank_30k_cross_topic_mace_p.csv"

RAW_COL_ARGUMENT = "argument"
RAW_COL_WA = "WA"
RAW_COL_MACE_P = "MACE-P"
RAW_COL_STANCE_WA = "stance_WA"
RAW_COL_STANCE_WA_CONF = "stance_WA_conf"

IN_TOPIC_SET_DISTRIBUTION_SEED = 17864564849235

logger = logging.getLogger(__file__)


def map_stance(stance):
    if stance == 1:
        return STANCE_PRO_LABEL
    if stance == -1:
        return STANCE_CON_LABEL

    raise ValueError(f"Unexpected value {stance} when mapping stance!")


def load_dataset(input_filepath):
    raw_dataset_path = os.path.join(input_filepath, DATASET_INPUT_FILE)
    raw_df = pd.read_csv(raw_dataset_path)
    return raw_df


def process_shared_cols(raw_df):
    raw_df.index.name = COL_ID
    raw_df[COL_IS_ARGUMENT] = True
    raw_df[COL_STANCE] = raw_df[RAW_COL_STANCE_WA].map(map_stance)
    return raw_df.rename(
        columns={
            RAW_COL_STANCE_WA_CONF: COL_STANCE_CONF,
            RAW_COL_ARGUMENT: COL_SENTENCE,
        }
    )


def process_mace_p_cols(mace_p_df):
    mace_p_df = mace_p_df.rename(columns={RAW_COL_MACE_P: COL_IS_STRONG_PROB})
    return adjust_dataset_format(mace_p_df)


def process_wa_cols(wa_df):
    wa_df = wa_df.rename(columns={RAW_COL_WA: COL_IS_STRONG_PROB})
    return adjust_dataset_format(wa_df)


def process(input_filepath, output_filepath):
    logger.info("Loading dataset ...")
    raw_df = load_dataset(input_filepath)
    logger.info("... done!")

    logger.info("Processing ...")
    # Shared
    shared_df = process_shared_cols(raw_df)
    topic_split_restriction_df = shared_df[[COL_TOPIC, COL_SET]].drop_duplicates()
    topic_split_restriction = dict(zip(topic_split_restriction_df[COL_TOPIC], topic_split_restriction_df[COL_SET]))
    if len(topic_split_restriction_df) != len(topic_split_restriction):
        raise AssertionError("Expected raw dataset to be cross-topic for ArgRank30k!")
    add_set_column(IN_TOPIC_SET_DISTRIBUTION_SEED, shared_df)

    # Specific metrics
    mace_p_df = shared_df
    wa_df = shared_df.copy()
    mace_p_df = process_mace_p_cols(mace_p_df)
    wa_df = process_wa_cols(wa_df)

    wa_new_restriction = in_and_cross_topic_to_output(
        df=wa_df,
        seed=None,
        topic_split_restriction=topic_split_restriction,
        output_filepath=output_filepath,
        in_topic_output=IN_TOPIC_DATASET_WA_OUTPUT_FILE,
        cross_topic_output=CROSS_TOPIC_DATASET_WA_OUTPUT_FILE,
    )
    logger.info("... WA done ...")
    mace_p_new_restriction = in_and_cross_topic_to_output(
        df=mace_p_df,
        seed=None,
        topic_split_restriction=wa_new_restriction,
        output_filepath=output_filepath,
        in_topic_output=IN_TOPIC_DATASET_MACE_P_OUTPUT_FILE,
        cross_topic_output=CROSS_TOPIC_DATASET_MACE_P_OUTPUT_FILE,
    )
    logger.info("... MACE-P done!")
    if wa_new_restriction != mace_p_new_restriction:
        raise AssertionError("Different topic assignment for WA/MACE-P!")

    return wa_new_restriction
