# -*- coding: utf-8 -*-
import click
import logging
import logging.config
from microscope.log_config import LOGGING


logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def main(input_filepath, output_file):
    """
    Runs specific data processing and aggregations to turn interim data
    from (../interim) into the final dataset.

    Examples / Ideas:
    - Extract columns from different files in interim data
    - Aggregate over grouped data
    """
    logger.info("Making final data set from interim data...")


if __name__ == "__main__":
    main()
