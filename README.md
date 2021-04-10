microscope
==============================

A short description of the project.

Project Organization
------------

```nohighlight
├── LICENSE
├── Makefile           <- Makefile with commands like `make data` or `make train`
├── README.md          <- The top-level README for developers using this project.
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
├── docs               <- A default Sphinx project; see sphinx-doc.org for details
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
│                         the creator's initials, and a short `-` delimited description, e.g.
│                         `1.0-jqp-initial-data-exploration`.
│
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   └── figures        <- Generated graphics and figures to be used in reporting
│
├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
│                         generated with `pip freeze > requirements.txt`
│
├── setup.py           <- Make this project pip installable with `pip install -e`
├── microscope                <- Source code for use in this project.
│   ├── __init__.py    <- Makes microscope a Python module
│   │
│   ├── data           <- Scripts to download or generate data and for generic transformations
│   │   └── generic_processing.py
│   │
│   ├── features       <- Scripts to turn raw data into features for modeling
│   │   └── build_features.py
│   │
│   ├── models         <- Scripts to train models and then use trained models to make
│   │   │                 predictions
│   │   ├── metric_config.py <- Contains the dictionary that configures the
│   │   │                       metrics used to evaluate your models
│   │   ├── model_config.py <- Contains the dictionary that configures the
│   │   │                       models to train and the target value
│   │   ├── predict_model.py
│   │   └── train_model.py
│   │
│   └── visualization  <- Scripts to create exploratory and results oriented visualizations
│       └── visualize.py
├── tests              <- Folder with initial tests for 100% coverage
└── tox.ini            <- tox file with settings for running tox; see tox.testrun.org
```


--------

<p><small>Project based on the <a target="_blank"
href="https://waveFrontSet.github.io/grip-on-data-science/">GriP on Data Science
template</a>.</small></p>
