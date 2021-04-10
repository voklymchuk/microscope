import pandas as pd
from functools import partial


read_file = partial(pd.read_csv)


def to_file(df, file_name, *args, **kwargs):
    """
    Writes the DataFrame `df` to a file in `file_name`.

    This is an example implementation that delegates to
    `DataFrame.to_csv` and freezes some standard arguments.
    When rewriting this and switching to a different file
    format, you need to rewrite test_train_model as well.
    """
    to_file_func = partial(df.to_csv, file_name, index=False)
    to_file_func(*args, **kwargs)
