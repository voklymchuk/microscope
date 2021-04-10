# -*- coding: utf-8 -*-
import click
import logging
import logging.config
from microscope.log_config import LOGGING


logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)


@click.command()
@click.argument("input_filepath", type=click.Path(exists=True))
@click.argument("output_filepath", type=click.Path())
def main(input_filepath, output_filepath):
    """
    Runs generic data processing and cleanup scripts to turn raw data
    from (../raw) into interim data that can be further analyzed and aggregated.

    Examples / Ideas:
    - Delete duplicate rows in raw data that are results of a bad SQL
    - Remove columns that are constant
    - Remove columns that only contain technical information and are not
    needed for aggregation (ids)
    """
    #logger.info("Running generic data processing and cleanup scripts...")
    print_hihi("VALVAL")


def print_hihi(name1):
    print("Hi-hi val ")
    print(name1)


if __name__ == "__main__":
    main()
