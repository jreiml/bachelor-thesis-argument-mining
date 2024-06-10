import os

import pandas as pd

from constants import COL_IS_STRONG_PROB, get_in_topic_dataset_name, get_cross_topic_dataset_name

METRIC_WA = "Weighted Average"
METRIC_MACE_P = "MACE Probability"
DATASET_NAME = "IBM-Rank-30k"
IN_TOPIC_DATASET_NAME = get_in_topic_dataset_name(DATASET_NAME)
CROSS_TOPIC_DATASET_NAME = get_cross_topic_dataset_name(DATASET_NAME)
IN_TOPIC_DATASET_FILE_WA = "arg_quality_rank_30k_wa.csv"
CROSS_TOPIC_DATASET_FILE_WA = "arg_quality_rank_30k_cross_topic_wa.csv"
IN_TOPIC_DATASET_FILE_MACE_P = "arg_quality_rank_30k_mace_p.csv"
CROSS_TOPIC_DATASET_FILE_MACE_P = "arg_quality_rank_30k_cross_topic_mace_p.csv"


def generate_is_arg_prob_histogram(df, metric_name, output_filepath):
    df_sent_len = df[COL_IS_STRONG_PROB]
    sent_len_hist = df_sent_len.hist()
    fig = sent_len_hist.get_figure()
    fig.suptitle(f"Argument strength of {DATASET_NAME} ({metric_name})")
    fig.supxlabel("Argument strength")
    fig.supylabel("Amount of sentences")
    metric_filename = metric_name.lower().replace(" ", "_")
    output_file = os.path.join(output_filepath, DATASET_NAME, f"argument_strength_{metric_filename}")
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    fig.clear()


def generate(input_filepath, output_filepath):
    input_file_wa = os.path.join(input_filepath, IN_TOPIC_DATASET_FILE_WA)
    df_wa = pd.read_csv(input_file_wa)
    input_file_mace_p = os.path.join(input_filepath, IN_TOPIC_DATASET_FILE_MACE_P)
    df_mace_p = pd.read_csv(input_file_mace_p)
    generate_is_arg_prob_histogram(df_wa, METRIC_WA, output_filepath)
    generate_is_arg_prob_histogram(df_mace_p, METRIC_MACE_P, output_filepath)
