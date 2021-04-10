Commands
========

The Makefile contains the central entry points for common tasks related to this project.

Data related commands
^^^^^^^^^^^^^^^^^^^^^

`data/raw`
~~~~~~~~~~
This target obtains raw data. This may be as simple as syncing data from S3 or
copying from another directory, or as hard as issuing complex SQL statements
or executing a web crawler.

**After implementing the target, document what exactly is being done here!**

`data/interim`
~~~~~~~~~~~~~~
This target creates generically processed and cleaned up data from raw data.

**After implementing the target, document what exactly is being done here!**

`data/processed`
~~~~~~~~~~~~~~~~
For the time being, this target merely creates the directory if it doesn't
exist.

`data/processed/dataset.csv`
~~~~~~~~~~~~~~~~~~~~~~~~~~~~
This target creates the final dataset from the interim data. Here,
transformations and aggregations specific to the project are applied.

**After implementing the target, document what exactly is being done here!**

`data`
~~~~~~
Alias for `data/processed/dataset.csv`. Additionally, this targets checks if all
requirements are installed.

Model related commands
^^^^^^^^^^^^^^^^^^^^^^

`train_test_split`
~~~~~~~~~~~~~~~~~~
Splits the final dataset `data/processed/dataset.csv` into a train set
`data/processed/train.csv` and a test set `data/processed/test.csv`.

`train`
~~~~~~~
The models specified in `microscope/models/model_config.py`
are fit to the train set `data/processed/train.csv` and saved in the folder `models/`.

`evaluate`
~~~~~~~~~~
For each model and associated metrics specified in
`microscope/models/metric_config.py`, the metrics are
evaluated using the test set `data/processed/test.csv`. The results are
persisted in `models/report.json`.

Notebook related commands
^^^^^^^^^^^^^^^^^^^^^^^^^
`clean_nb_%`
~~~~~~~~~~~~
Clears the output of the jupyter notebook in the folder `notebooks/` whole
filename (without the extension `ipynb`) is `%`. For instance, `make
clean_nb_Untitled` clears the output of the notebook `notebooks/Untitled.ipynb`.

`cleanall_nb`
~~~~~~~~~~~~~
Clears the output of all jupyter notebooks in `notebooks/`.

Syncing data to S3
^^^^^^^^^^^^^^^^^^

