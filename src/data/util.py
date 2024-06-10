import os

import numpy as np
from numpy.random import RandomState, MT19937, SeedSequence

from constants import (
    PROCESSED_DATASET_FORMAT,
    COL_SET,
    SET_TRAIN,
    SET_DEV,
    SET_TEST,
    STANCE_CON_LABEL,
    STANCE_PRO_LABEL,
    STANCE_NO_ARG_LABEL,
    TRAIN_DATASET_SIZE,
    DEV_DATASET_SIZE,
    COL_TOPIC,
)


def adjust_dataset_format(df):
    dataset_format = [col for col in PROCESSED_DATASET_FORMAT if col in df.columns]
    return df[dataset_format]


def get_set_label_for_number(number):
    if number < TRAIN_DATASET_SIZE:
        return SET_TRAIN

    if number < TRAIN_DATASET_SIZE + DEV_DATASET_SIZE:
        return SET_DEV

    return SET_TEST


def create_set_column(seed, row_count):
    rs = RandomState(MT19937(SeedSequence(seed)))
    numbers = rs.rand(row_count, 1)
    convert_fn = np.vectorize(get_set_label_for_number)
    set_col = convert_fn(numbers)
    return set_col


def set_split_by_topic(df, seed=None, topic_split_restriction=None):
    cross_topic_df = df.copy()
    new_restriction = dict() if topic_split_restriction is None else dict(topic_split_restriction)
    rs = None if seed is None else RandomState(MT19937(SeedSequence(seed)))

    for topic in sorted(set(cross_topic_df[COL_TOPIC])):
        if topic not in topic_split_restriction:
            if seed is None:
                raise ValueError("Seed is None, but not all topics were specified in topic_split_restriction!")
            new_restriction[topic] = get_set_label_for_number(rs.random())
        cross_topic_df.loc[cross_topic_df[COL_TOPIC] == topic, COL_SET] = new_restriction[topic]

    return cross_topic_df, new_restriction


def in_and_cross_topic_to_output(
    df, seed, topic_split_restriction, output_filepath, in_topic_output, cross_topic_output
):
    in_topic = os.path.join(output_filepath, in_topic_output)
    df.to_csv(in_topic)

    cross_topic_df, new_restriction = set_split_by_topic(df, seed, topic_split_restriction)
    cross_topic = os.path.join(output_filepath, cross_topic_output)
    cross_topic_df.to_csv(cross_topic)

    return new_restriction


def add_set_column(seed, df):
    set_col = create_set_column(seed, len(df))
    df[COL_SET] = set_col


def invert_stance(stance):
    if stance == STANCE_NO_ARG_LABEL:
        return STANCE_NO_ARG_LABEL

    if stance == STANCE_CON_LABEL:
        return STANCE_PRO_LABEL

    if stance == STANCE_PRO_LABEL:
        return STANCE_CON_LABEL

    raise ValueError(f"Unexpected value {stance} when inverting stance!")
