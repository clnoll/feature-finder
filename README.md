#### Feature Finder

To execute this program, download the feature-finder files and run the following from your command line:

```
virtualenv -p python3 venv; . ./venv/bin/activate
python3 setup.py install
feature-finder <model type> <path/to/data.csv> <index or name of y-column> <optional: --header>
```

For example:
```
feature-finder logistic feature_finder/tests/sample_data/01_feature_logistic.csv 1
```

The following help menu is available via `feature-finder --help`.
```
usage: feature-finder [-h] [--header] [--plugins PLUGINS [PLUGINS ...]]
                      model data y

positional arguments:
  model                 Model type: linear or logistic.
  data                  Data in CSV format.
  y                     y-value of column (index position or header name).

optional arguments:
  -h, --help            show this help message and exit
  --header              Data CSV contains header row.
  --plugins PLUGINS [PLUGINS ...]
                        Use plugin(s) defined in plugins directory to add or
                        clean up feature columns.
```

Your CSV should include at least one x-column with numerical data for input as a dependent variable into themodel, and one y-column with either quantitative or categorical data.
e.g.
```
1, 22.2, 30
0, 20.5, 28
0, 30, 33.5
```

Plugins are customizations that can be added to the `plugins/plugins.py` file. A sample plugin has been included for reference. Plugins must be python functions that take in a pandas DataFrame object of the original CSV, which it is allowed to modify.

The plugin must return a pandas DataFrame that includes the features whose error/accuracy should be calculated.

Modify your data CSV with plugins using the following input format:
```
feature-finder <model type> <path/to/data.csv> <index or name of y-column> <optional: --header> --plugins <plugin1 plugin2 etc>
```

Tests can be run from the `feature-finder` directory with `nosetests .`.
