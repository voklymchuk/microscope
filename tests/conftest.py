import pandas as pd
import pytest
from click.testing import CliRunner
from microscope.models.model_config import TARGET_NAME


@pytest.fixture
def runner():
    runner = CliRunner()
    with runner.isolated_filesystem():
        yield runner


@pytest.fixture
def df():
    return pd.DataFrame({"OBJECTID": [0, 1, 2, 3, 4], TARGET_NAME: [1, 1, 3, 4, 1]})
