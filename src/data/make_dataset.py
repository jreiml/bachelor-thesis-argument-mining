# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
from dotenv import find_dotenv, load_dotenv

import arg_quality_rank_30k
import evidences_sentences
import ukp_sentential_argument_mining


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """Runs data processing scripts to turn raw data from (../raw) into
    cleaned data ready to be analyzed (saved in ../processed).
    """
    logger = logging.getLogger(__file__)
    logger.info("making final data set from raw data")
    # Start with ArgRank30k, since it is cross-topic by default
    topic_split_restriction = arg_quality_rank_30k.process(input_filepath, output_filepath)
    topic_split_restriction = ukp_sentential_argument_mining.process(
        input_filepath, output_filepath, topic_split_restriction
    )
    evidences_sentences.process(input_filepath, output_filepath, topic_split_restriction)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
