import os

import pandas as pd

from constants import COL_STANCE, COL_IS_ARGUMENT, get_in_topic_dataset_name, get_cross_topic_dataset_name

DATASET_NAME = "UKP Sentential"
IN_TOPIC_DATASET_NAME = get_in_topic_dataset_name(DATASET_NAME)
CROSS_TOPIC_DATASET_NAME = get_cross_topic_dataset_name(DATASET_NAME)
IN_TOPIC_DATASET_FILE = "ukp_sentential_argument_mining.csv"
CROSS_TOPIC_DATASET_FILE = "ukp_sentential_argument_mining_cross_topic.csv"


def generate_stance_bar(df, output_filepath):
    stance_counts = df[COL_STANCE].value_counts()
    stance_bar = stance_counts.plot(kind="bar", rot=0)
    fig = stance_bar.get_figure()
    fig.suptitle(f"Stance distribution for {DATASET_NAME}")
    fig.supylabel("Amount of sentences")
    fig.supxlabel("Stance label")
    output_file = os.path.join(output_filepath, DATASET_NAME, "stance")
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    fig.clear()


def generate_is_arg_bar(df, output_filepath):
    is_arg_counts = df[COL_IS_ARGUMENT].value_counts()
    is_arg_bar = is_arg_counts.plot(kind="bar", rot=0)
    fig = is_arg_bar.get_figure()
    fig.suptitle(f"Argument distribution for {DATASET_NAME}")
    fig.supylabel("Amount of sentences")
    fig.supxlabel("Argument label")
    output_file = os.path.join(output_filepath, DATASET_NAME, "is_arg")
    fig.tight_layout()
    fig.savefig(output_file, bbox_inches="tight")
    fig.clear()


def generate(input_filepath, output_filepath):
    input_file = os.path.join(input_filepath, IN_TOPIC_DATASET_FILE)
    df = pd.read_csv(input_file)
    generate_is_arg_bar(df, output_filepath)
    generate_stance_bar(df, output_filepath)
