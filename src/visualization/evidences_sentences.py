import os

import pandas as pd

from constants import COL_IS_ARGUMENT_PROB, get_in_topic_dataset_name, get_cross_topic_dataset_name

DATASET_NAME = "IBM Evidences Sentences"
IN_TOPIC_DATASET_NAME = get_in_topic_dataset_name(DATASET_NAME)
CROSS_TOPIC_DATASET_NAME = get_cross_topic_dataset_name(DATASET_NAME)
IN_TOPIC_DATASET_FILE = "evidences_sentences.csv"
CROSS_TOPIC_DATASET_FILE = "evidences_sentences_cross_topic.csv"
ACCEPTANCE_RATE_ALPHAS_SUBFOLDER = "acceptance_rate_alphas"

ACCEPTANCE_RATE_ALPHAS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def generate_is_arg_prob_histogram(df, output_filepath):
    df_sent_len = df[COL_IS_ARGUMENT_PROB]
    sent_len_hist = df_sent_len.hist()
    fig = sent_len_hist.get_figure()
    fig.suptitle(f"Argument acceptance rate of {DATASET_NAME}")
    fig.supxlabel("Acceptance rate")
    fig.supylabel("Amount of sentences")
    output_file = os.path.join(output_filepath, DATASET_NAME, "acceptance_rate")
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    fig.clear()


def generate_is_arg_bar_for_alphas(df, output_filepath):
    directory = os.path.join(output_filepath, DATASET_NAME, ACCEPTANCE_RATE_ALPHAS_SUBFOLDER)
    if not os.path.exists(directory):
        os.makedirs(directory)

    for i, alpha in enumerate(ACCEPTANCE_RATE_ALPHAS):
        df_alpha = df[COL_IS_ARGUMENT_PROB].map(lambda x: x > alpha)
        is_arg_counts = df_alpha.value_counts()
        is_arg_bar = is_arg_counts.plot(kind="bar", rot=0)
        fig = is_arg_bar.get_figure()
        fig.suptitle(f"Argument distribution for {DATASET_NAME} with alpha {alpha}")
        fig.supylabel("Amount of sentences")
        fig.supxlabel("Argument label")
        output_file = os.path.join(directory, f"alpha_{i}")
        fig.tight_layout()
        fig.savefig(output_file, bbox_inches="tight")
        fig.clear()


def generate(input_filepath, output_filepath):
    input_file = os.path.join(input_filepath, IN_TOPIC_DATASET_FILE)
    df = pd.read_csv(input_file)
    generate_is_arg_prob_histogram(df, output_filepath)
    generate_is_arg_bar_for_alphas(df, output_filepath)
