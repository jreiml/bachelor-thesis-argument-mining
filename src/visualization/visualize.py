# -*- coding: utf-8 -*-
import os

import click
import logging
from pathlib import Path

import matplotlib
from dotenv import find_dotenv, load_dotenv

import arg_quality_rank_30k
import evidences_sentences
import ukp_sentential_argument_mining
from constants import DEFAULT_FONT_SIZE
import shared

DATASET_NAMES = [
    arg_quality_rank_30k.DATASET_NAME,
    arg_quality_rank_30k.IN_TOPIC_DATASET_NAME,
    arg_quality_rank_30k.CROSS_TOPIC_DATASET_NAME,
    evidences_sentences.DATASET_NAME,
    evidences_sentences.IN_TOPIC_DATASET_NAME,
    evidences_sentences.CROSS_TOPIC_DATASET_NAME,
    ukp_sentential_argument_mining.DATASET_NAME,
    ukp_sentential_argument_mining.IN_TOPIC_DATASET_NAME,
    ukp_sentential_argument_mining.CROSS_TOPIC_DATASET_NAME,
]


def create_output_dirs(output_filepath):
    for dataset in DATASET_NAMES:
        directory = os.path.join(output_filepath, dataset)
        if not os.path.exists(directory):
            os.makedirs(directory)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn processed data into figures"""
    logger = logging.getLogger(__file__)
    logger.info("creating visualizations from final datasets")
    create_output_dirs(output_filepath)
    matplotlib.rcParams.update({"font.size": DEFAULT_FONT_SIZE})
    shared.generate(input_filepath, output_filepath)
    arg_quality_rank_30k.generate(input_filepath, output_filepath)
    evidences_sentences.generate(input_filepath, output_filepath)
    ukp_sentential_argument_mining.generate(input_filepath, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
