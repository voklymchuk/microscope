from microscope.models.train_model import main
from microscope.utils import to_file


def test_split(runner, df):
    to_file(df, "dataset.csv")
    result = runner.invoke(main, ["split", "dataset.csv", "."])
    assert result.exit_code == 0


def test_train(runner, df):
    to_file(df, "dataset.csv")
    result = runner.invoke(main, ["train", "dataset.csv", "."])
    assert result.exit_code == 0
