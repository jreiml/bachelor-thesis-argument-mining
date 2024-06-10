import glob
import json
import logging
from pathlib import Path

import click
from coverage.collector import os
from dotenv import load_dotenv, find_dotenv

from utils import average_results


def average_runs(input_filepath_filter, output_filepath):
    results = []
    for result_path in glob.glob(input_filepath_filter, recursive=True):
        if not os.path.isfile(result_path):
            continue

        with open(result_path, "r") as result_file:
            results.append(json.load(result_file))

    average = average_results(results)
    if len(average) > 0:
        with open(output_filepath, "w") as average_results_file:
            json.dump(average, average_results_file, indent=4)


@click.command()
@click.argument("input_filepath_filter", type=click.STRING)
@click.argument("output_filepath", type=click.Path(dir_okay=False))
def main(input_filepath_filter, output_filepath):
    """Runs scripts to process results"""
    logger = logging.getLogger(__file__)
    logger.info(f"processing results for {input_filepath_filter} to {output_filepath}")
    average_runs(input_filepath_filter, output_filepath)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
