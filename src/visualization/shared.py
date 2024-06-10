import logging
import os

import pandas as pd

import arg_quality_rank_30k
import evidences_sentences
import ukp_sentential_argument_mining
from constants import COL_SENTENCE, COL_TOPIC, COL_SET, SET_TRAIN, SET_DEV, SET_TEST

logger = logging.getLogger(__file__)

DATASET_FILES = [
    (
        arg_quality_rank_30k.DATASET_NAME,
        arg_quality_rank_30k.IN_TOPIC_DATASET_FILE_WA,
    ),
    (
        evidences_sentences.DATASET_NAME,
        evidences_sentences.IN_TOPIC_DATASET_FILE,
    ),
    (
        ukp_sentential_argument_mining.DATASET_NAME,
        ukp_sentential_argument_mining.IN_TOPIC_DATASET_FILE,
    ),
]

DATASET_FILES_WITH_SPLIT_DIFFERENCES = [
    (
        arg_quality_rank_30k.IN_TOPIC_DATASET_NAME,
        arg_quality_rank_30k.IN_TOPIC_DATASET_FILE_WA,
    ),
    (
        arg_quality_rank_30k.CROSS_TOPIC_DATASET_NAME,
        arg_quality_rank_30k.CROSS_TOPIC_DATASET_FILE_WA,
    ),
    (
        evidences_sentences.IN_TOPIC_DATASET_NAME,
        evidences_sentences.IN_TOPIC_DATASET_FILE,
    ),
    (
        evidences_sentences.CROSS_TOPIC_DATASET_NAME,
        evidences_sentences.CROSS_TOPIC_DATASET_FILE,
    ),
    (
        ukp_sentential_argument_mining.IN_TOPIC_DATASET_NAME,
        ukp_sentential_argument_mining.IN_TOPIC_DATASET_FILE,
    ),
    (
        ukp_sentential_argument_mining.CROSS_TOPIC_DATASET_NAME,
        ukp_sentential_argument_mining.CROSS_TOPIC_DATASET_FILE,
    ),
]

TOPIC_COUNT = 10


def generate_length_histogram(df, name, output_filepath, split=None, quantile=1.0):
    filtered_df = df if split is None else df.loc[df[COL_SET] == split]
    max_len = max(df[COL_SENTENCE].map(len))
    df_sent_len = filtered_df[COL_SENTENCE].map(len)
    if quantile < 1:
        adjusted_max_len = df_sent_len.quantile(0.99)
        df_sent_len = df_sent_len.map(lambda x: min(adjusted_max_len, x))
    sent_len_hist = df_sent_len.hist(range=(0, max_len))
    fig = sent_len_hist.get_figure()
    quantile_name = int(quantile * 100)
    quantile_title = "" if quantile >= 1.0 else f" ({quantile_name}th percentile)"
    split_title = "" if split is None else f" (Split: {split})"
    fig.suptitle(f"Sentence length of {name}{split_title}{quantile_title}")
    fig.supxlabel("Sentence length")
    fig.supylabel("Amount of sentences")
    output_filename = (
        f"sentence_length_{quantile_name}.png" if split is None else f"sentence_length_{split}_{quantile_name}.png"
    )
    output_file = os.path.join(output_filepath, name, output_filename)
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    fig.clear()


def generate_topic_hbar(df, name, output_filepath):
    topic_counts = df[COL_TOPIC].value_counts()
    total_topics = len(topic_counts)
    top_topics = topic_counts.head(TOPIC_COUNT)
    topic_bar = top_topics.plot(kind="barh")
    fig = topic_bar.get_figure()
    actual_topics = min(total_topics, TOPIC_COUNT)
    fig.suptitle(f"Top {actual_topics}/{total_topics} topics for {name}")
    fig.supxlabel("Amount of sentences for topic")
    output_file = os.path.join(output_filepath, name, "topics")
    fig.savefig(output_file, bbox_inches="tight")
    fig.clear()


def generate_set_split_bar(df, name, output_filepath):
    split_counts = df[COL_SET].value_counts()
    split_bar = split_counts.plot(kind="bar", rot=0)
    fig = split_bar.get_figure()
    fig.suptitle(f"Dataset split for {name}")
    fig.supylabel("Amount of sentences")
    fig.supxlabel("Split")
    output_file = os.path.join(output_filepath, name, "set_split")
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    fig.clear()


def generate(input_filepath, output_filepath):
    for name, filename in DATASET_FILES:
        input_file = os.path.join(input_filepath, filename)
        df = pd.read_csv(input_file)
        generate_length_histogram(df, name, output_filepath, None, 1.0)
        generate_length_histogram(df, name, output_filepath, None, 0.99)
        generate_topic_hbar(df, name, output_filepath)

    for name, filename in DATASET_FILES_WITH_SPLIT_DIFFERENCES:
        input_file = os.path.join(input_filepath, filename)
        df = pd.read_csv(input_file)
        generate_set_split_bar(df, name, output_filepath)
        generate_length_histogram(df, name, output_filepath, SET_TRAIN, 1.0)
        generate_length_histogram(df, name, output_filepath, SET_DEV, 1.0)
        generate_length_histogram(df, name, output_filepath, SET_TEST, 1.0)
        generate_length_histogram(df, name, output_filepath, SET_TRAIN, 0.99)
        generate_length_histogram(df, name, output_filepath, SET_DEV, 0.99)
        generate_length_histogram(df, name, output_filepath, SET_TEST, 0.99)
