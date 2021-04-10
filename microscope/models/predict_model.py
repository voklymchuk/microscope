# -*- coding: utf-8 -*-
import click
import logging
import logging.config
import json
from pathlib import Path
from joblib import load
from microscope.models.model_config import TARGET_NAME
from microscope.models.metric_config import METRICS
from microscope.log_config import LOGGING
from microscope.utils import read_file


logging.config.dictConfig(LOGGING)
logger = logging.getLogger(__name__)


@click.group()
def main():
    pass


@main.command()
@click.argument("test_file", type=click.Path(exists=True))
@click.argument("model_path", type=click.Path(exists=True))
@click.argument("output_file", type=click.Path())
def evaluate(test_file, model_path, output_file):
    logger.info("Start evaluating %d models...", len(METRICS.keys()))
    test = read_file(test_file)
    X, y = test.drop(columns=[TARGET_NAME]), test[TARGET_NAME]
    model_pathobj = Path(model_path)
    metrics_per_model = {}
    for name, metrics in METRICS.items():
        logger.debug("Evaluating model %s", name)
        model = load(model_pathobj / name)
        y_pred = model.predict(X)
        metric_results = {metric.__name__: metric(y, y_pred) for metric in metrics}
        logger.info("Metrics for model %s: %s", name, metric_results)
        metrics_per_model[name] = metric_results
    with open(output_file, "w+") as f:
        json.dump(metrics_per_model, f, indent=2)


if __name__ == "__main__":
    main()
