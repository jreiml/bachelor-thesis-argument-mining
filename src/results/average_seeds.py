import glob
import json
import logging
import pathlib
import re
from collections import defaultdict
from pathlib import Path

import click
from coverage.collector import os
from dotenv import load_dotenv, find_dotenv

from utils import average_results


def add_average_results(result_files, average_seed_path, result_path):
    parsed_average_seed_path = pathlib.Path(average_seed_path)
    average_result_path = os.path.join(*parsed_average_seed_path.parts[:4], "average.json")
    result_files[average_result_path].append(result_path)

    def add_specific_result(filter, name):
        if filter in result_path:
            average_specific_result_path = os.path.join(*parsed_average_seed_path.parts[:4], f"average_{name}.json")
            result_files[average_specific_result_path].append(result_path)

    add_specific_result("-no_motion-", "no_motion")
    add_specific_result("-motion-", "motion")
    add_specific_result("in_topic", "in_topic")
    add_specific_result("cross_topic", "cross_topic")


def get_result_files_mapping(input_run_directories, output_filepath):
    result_files = defaultdict(list)
    for input_run_directory in input_run_directories:
        for result_path in glob.glob(f"{input_run_directory}/**/*.json", recursive=True):
            if "average" in result_path:
                raise NotImplementedError("Average of averages not supported!")
            if not os.path.isfile(result_path):
                continue

            average_seed_path = re.sub(r"results(/)+[0-9]+", output_filepath, result_path)
            average_seed_dir = os.path.dirname(average_seed_path)
            if not os.path.exists(average_seed_dir):
                os.makedirs(average_seed_dir)
            result_files[average_seed_path].append(result_path)
            add_average_results(result_files, average_seed_path, result_path)

    return result_files


def output_results(result_files):
    for output_result_filepath, input_result_filepaths in result_files.items():
        results = []
        for input_result_filepath in input_result_filepaths:
            with open(input_result_filepath, "r") as input_result_file:
                input_result = json.load(input_result_file)
                results.append(input_result)
        average_result = average_results(results)

        if len(average_result) > 0:
            with open(output_result_filepath, "w") as output_result_file:
                json.dump(average_result, output_result_file, indent=4)


def output_input_file_logs(result_files, output_filepath):
    input_files_log_path = os.path.join(output_filepath, "input_files.json")
    with open(input_files_log_path, "w") as input_files_log:
        json.dump(result_files, input_files_log, indent=4)


def average_seeds(input_run_directories, output_filepath):
    result_files = get_result_files_mapping(input_run_directories, output_filepath)
    output_results(result_files)
    output_input_file_logs(result_files, output_filepath)


@click.command()
@click.argument("input_run_directories", nargs=-1, type=click.STRING)
@click.argument("output_directory", nargs=1, type=click.Path(dir_okay=True))
def main(input_run_directories, output_directory):
    """Runs scripts to average runs across different seeds"""
    logger = logging.getLogger(__file__)
    logger.info(f"averaging runs for {input_run_directories} to {output_directory}")
    average_seeds(input_run_directories, output_directory)


if __name__ == "__main__":
    log_fmt = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    load_dotenv(find_dotenv())

    main()
