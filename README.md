# Design

The machine pipeline comprises 3 stages: **Extract**, **Preprocess**, and **Evaluate**. 

## Extract
- This stage ingests the data from the SQL database provided
- The connection and query details can be modified through the `config.yaml` file

## Preprocess
- This stage cleans the data and augments it with additional features.
- Knowing what to clean up and what features to add was done through an exploratory data anlaysis process. This is documented in the `eda.ipynb` notebook
- In this stage, we also split the data into training and test sets in preparation for the next stage

## Evaluate
- In this stage, we fit the models to the input data (extracted and preprocessed)
- This stage makes heavy use of the [scikit-learn API](https://scikit-learn.org/stable/modules/classes.html)
- The choice of model, scaler (within the `sklearn` library) and their associated parameters can be adjusted through the `config.yaml` file
- The default model is set to `ensemble.RandomForestRegressor`. This model was chosen as it is robust to overfitting, and achieved the best cross-validation score during early testing. However, this can be easily changed in `config.yaml`.
- This stage makes use of the `GridSearchCV` implemented in `sklearn` to search over the hyperparameter values to find combination which achieves the best cross-validation scores.
- The hyperparameter search space can be specified in `config.yaml`
- By default the model is not evaluated on the test set. This can be changed through the command line argument `-t` (described below). However, the user should be wary of doing this prematurely and inadvertently snooping on test data.

# Usage
- This pipeline was designed to facilitate adjust-evaluate iterative cycle, and to be highly configurable.
- The settings can either be changed through the `config.yaml` file, or through command line arguments (described below).

## The `config.yaml` file
- In this file, settings for all the stages described above can be modified
- The most important section is `ml_config`, which selects the model to be used, as well as the hyperparameter search space to conduct `GridSearchCV` over
- The `yaml` specification can be found [here](https://yaml.org/spec/) 

## Command line arguments:

```
usage: run.sh [-h] [-t] [-l] [-r [N] | -p [N]]

optional arguments:
  -h, --help            show this help message and exit
  -t, --test            Evaluate test set and generate evaluation metrics
  -l, --log             Log params and results in log.json. If such a file does not exist, it is created.
  -r [N], --peek_raw [N]
                        Preview raw data extracted from database (part 1 of assessment). Display first N rows (default 5)
  -p [N], --peek_training [N]
                        Preview training data that has been cleaned and augmented with extra features. Display first N rows (default 5)
```

- The command line arguments serve two main functions:
- The first function is to **select the mode of operation** of the script. The default mode is to run the full pipeline: extract, preprocess, and evaluate.
- If the `--peek_raw` mode is selected, the data is extracted and `N` rows are displayed (`N` passed in as command line argument)
- If the `--peek_training` mode is selected, the data is extracted, preprocessed and split into training and test sets. `N` rows of the training set are displayed (`N` passed in as command line argument)
- The second function of the command line arguments is to **modify the behaviour of the pipeline**
- Each time the pipeline is run, the evaluation metrics are printed to the command line. To persist the scores and parameters, use the `-l` flag, which appends the results and parameters in a `log.json` file.
- To evaluate the model on the test set, select the `-t` option. This runs the model on the test set and generates the relavant evaluation metrics. This option should only be selected at the last stage, as tuning the parameters too closely to the test set will undermine the test metrics.