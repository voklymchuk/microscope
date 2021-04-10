# -*- coding: utf-8 -*-
import click
import logging
import logging.config
from pathlib import Path
from sklearn.model_selection import train_test_split
from joblib import dump
from microscope.models.model_config import TARGET_NAME, MODELS
from microscope.log_config import LOGGING
from microscope.utils import read_file, to_file


logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command()
@click.argument("input_file", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def split(input_file, output_filepath):
    """
    Splits the `input_file` into a train set and a test set that
    are written to `output_filepath`.
    """
    logger.info("Splitting into train and test set...")
    output_dir = Path(output_filepath)
    dataset = read_file(input_file)
    logger.info("Dataset has %d lines.", len(dataset.index))
    train, test = train_test_split(dataset)
    logger.info("Train set has %d lines.", len(train.index))
    logger.info("Test set has %d lines.", len(test.index))
    to_file(train, output_dir / "train.csv")
    to_file(test, output_dir / "test.csv")


@main.command()
@click.argument("train_file", type=click.Path(exists=True))
@click.argument("output_path", type=click.Path())
def train(train_file, output_path):
    logger.info("Start training %d models...", len(MODELS.keys()))
    train = read_file(train_file)
    X, y = train.drop(columns=[TARGET_NAME]), train[TARGET_NAME]
    output_pathobj = Path(output_path)
    for name, model in MODELS.items():
        logger.debug("Training model %s", name)
        model.fit(X, y)
        logger.info("Score for model %s: %d", name, model.score(X, y))
        dump(model, output_pathobj / name)


if __name__ == "__main__":
    main()
