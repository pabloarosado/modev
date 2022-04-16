# Modev : Model Development for Data Science Projects

## Introduction

Most data science projects involve similar ingredients (loading data, defining some evaluation metrics, splitting data
into different train/validation/test sets, etc.).
Modev's goal is to ease these repetitive steps, without constraining the freedom data scientists need to develop models.

## Installation

The easiest is to install from pip:

```
pip install modev
```

Otherwise you can clone the latest release and install it manually:
```
git clone git@github.com:pabloarosado/modev.git
cd modev
python setup.py install
```

Otherwise you can install from conda:
```
conda install -c pablorosado modev
```

## Quick guide

The quickest way to get started with modev is to run a pipeline with the default settings:
```
import modev
pipe = modev.Pipeline()
pipe.run()
``` 
This runs a pipeline on some example data, and returns a dataframe with a ranking of approaches that perform best (given
some metrics) on the data.

To get the data used in the pipeline:
```
pipe.get_data()
```

By default, modev splits the data into a playground and a test set.
The test set is omitted (unless parameter execution_inputs['test_mode'] is set to True), and the playground is split
into k train/dev folds, to do k-fold cross-validation.
To get the indexes of train/dev/test sets:
```
pipe.get_indexes()
```

The pipeline will load two dummy approaches (which can be accessed on ```pipe.approaches_function```) with some
parameters (which can be accessed on ```pipe.approaches_pars```).
For each fold, these approaches will be fitted to the train set and predict the 'color' of the examples on the dev sets.

The metrics used to evaluate the performance of the approaches are listed in ```pipe.evaluation_pars['metrics']```.

An exhaustive grid search is performed, to get all possible combinations of the parameters of each of the approaches. 
The performance of each of these combinations on each fold can be accessed on:
```
pipe.get_results()
```

To plot these results per fold for each of the metrics:
```
pipe.plot_results()
```
To plot only a certain list of metrics, this list can be given as an argument of this function.

To get the final ranking of best approaches (after combining the results of different folds):
```
pipe.get_selected_models()
```

## Guide

The inputs accepted by ```modev.Pipeline``` refer to the usual ingredients in a data science project (data loading,
evaluation metrics, model selection method...).
We define an **experiment** as a combination of all these ingredients.
An experiment is defined by a dictionary with the following keys:
1. `load_inputs`: Dictionary of inputs related to data loading:
    + <details>
          <summary>Using the default function.</summary>

      If `function` is not given, `modev.etl.load_local_file` will be used. <br>
      This function loads a local (.csv) file. It uses `pandas.read_csv` function and accepts all its arguments, and also some additional arguments.
      * **Arguments that must be defined in `load_inputs`**:
          * `data_file` : str <br>
              Path to local (.csv) file. <br>
      * **Arguments that can optionally be defined in `load_inputs`**:
          * `selection` : str or None <br>
              Selection to perform on the data. For example, if selection is `"(data['height'] > 3) & (data['width'] < 2)"`,
              that selection will be evaluated and applied to the data; None to apply no selection. <br>
              Default: None
          * `sample_nrows` : int or None <br>
              Number of random rows to sample from the data (without repeating rows); None to load all rows. <br>
              Default: None
          * `random_state` : int or None <br>
              Random state (relevant only when sampling from data, i.e. when `sample_nrows` is not None). <br>
              Default: None
      </details>

    + <details>
          <summary>Using a custom function.</summary>

      If the `function` key is contained in the `load_inputs` dictionary, its value must be a valid function.
      * **Arguments that this custom function must accept**: <br>
          This function can have an arbitrary number of mandatory arguments (or none), to be specified in `load_inputs`.
      * Additionally, this function can have an arbitrary number of optional arguments (or none), to be specified in `load_inputs` dictionary.
      * **Outputs this custom function must return**: <br>
          * `data` : pd.DataFrame <br>
              Relevant data.
      </details>

2. `validation_inputs`: Dictionary of inputs related to validation method (e.g. k-fold or temporal-fold
    cross-validation).
    + <details>
          <summary>Using the default function.</summary>

      If `function` is not given, `modev.validation.k_fold_playground_n_tests_split` will be used. <br>
      This function generates indexes that split data into a playground (with k folds) and n test sets.
      There is only one playground, which contains train and dev sets, and has no overlap with test sets.
      Playground is split into k folds, namely k non-overlapping dev sets, and k overlapping train sets.
      Each of the folds contains all data in the playground (part of it in train, and the rest in dev); hence train and dev sets of the same fold do not overlap.
      * **Arguments that must be defined in `validation_inputs`**: <br>
          None (all arguments will be taken from default if not explicitly given).
      * **Arguments that can optionally be defined in `validation_inputs`**:
          * `playground_n_folds` : int <br>
            Number of folds to split playground into (also called `k`), so that there will be k train sets and k dev sets. <br>
            Default: 4
          * `test_fraction` : float <br>
            Fraction of data to use for test sets. <br>
            Default: 0.2
          * `test_n_sets` : int <br>
            Number of test sets. <br>
            Default: 2
          * `labels` : list or None <br>
            Labels to stratify data according to their distribution; None to not stratify data. <br>
            Default: None
          * `shuffle` : bool <br>
            True to shuffle data before splitting; False to keep them sorted as they are before splitting. <br>
            Default: True
          * `random_state` : int or None <br>
            Random state for shuffling; Ignored if 'shuffle' is False (in which case, 'random_state' can be set to None). <br>
            Default: None
          * `test_mode` : bool <br>
            True to return indexes of the test set; False to return indexes of the dev set. <br>
            Default: False
      </details>

    + <details>
          <summary>Using a custom function.</summary>

      If the `function` key is contained in the `validation_inputs` dictionary, its value must be a valid function.
      * **Arguments that this custom function must accept**:<br>
          * `data` : pd.DataFrame<br>
            Indexed data (e.g. a dataframe whose index can be accessed with `data.index`).
      * Additionally, this function can have an arbitrary number of optional arguments (or none), to be specified in `validation_inputs` dictionary.
      * **Outputs this custom function must return**: <br>
          * `train_indexes` : dict
              Indexes to use for training on the different k folds, e.g. for 10 folds: <br>
              `{0: np.array([...]), 1: np.array([...]), ..., 10: np.array([...])}` <br>
          * `test_indexes` : dict
              Indexes to use for evaluating (either dev or test) on the different k folds, e.g. for 10 folds and if test_mode is False: <br>
              `{0: np.array([...]), 1: np.array([...]), ..., 10: np.array([...])}`
      </details>

3. `execution_inputs`: Dictionary of inputs related to the execution of approaches.
    + <details>
          <summary>Using the default function.</summary>

      If `function` is not given, `modev.execution.execute_model` will be used.
      This function defines the execution method (including training and prediction, and any possible preprocessing) for an approach.
      This function takes an approach `approach_function` with parameters `approach_pars`, a train set (with predictors `train_x` and targets `train_y`) and the predictors of a test set `test_x`, and returns the predicted targets of the test set. <br>
      Note: Here, `test` refers to either a dev or a test set indistinctly.
      * **Arguments that must be defined in `execution_inputs`**:
          * `target` : str <br>
              Name of target column in both train_set and test_set.
      * **Arguments that can optionally be defined in `execution_inputs`**: <br>
          None (this function does not accept any other optional arguments).
      </details>

    + <details>
          <summary>Using a custom function.</summary>

      If the `function` key is contained in the `execution_inputs` dictionary, its value must be a valid function.
      * **Arguments that this custom function must accept**:<br>
          * `model` : model object <br>
              Instantiated approach.
          * `data` : pd.DataFrame <br>
              Data, as returned by load inputs function.
          * `fold_train_indexes` : np.array <br>
              Indexes of train set (or playground set) for current fold.
          * `fold_test_indexes` : np.array <br>
              Indexes of dev set (or test set) for current fold.
          * `target` : str <br>
              Name of target column in both train_set and test_set.
      * Additionally, this function can have an arbitrary number of optional arguments (or none), to be specified in `execution_inputs` dictionary.
      * **Outputs this custom function must return**: <br>
          * `execution_results` : dict <br>
              Execution results. It contains: <br>
              * `truth`: np.array of true values of the target in the dev (or test) set. <br>
              * `prediction`: np.array of predicted values of the target in the dev (or test) set.
      </details>

4. `evaluation_inputs`: Dictionary of inputs related to evaluation metrics.
    + <details>
          <summary>Using the default function.</summary>

      If `function` is not given, `modev.evaluation.evaluate_predictions` will be used. <br>
      This function evaluates predictions, given a ground truth, using a list of metrics.
      * **Arguments that must be defined in `evaluation_inputs`**:
          * metrics : list <br>
              Metrics to use for evaluation. Implemented methods include:
              * `precision`: usual precision in classification problems.
              * `recall`: usual recall in classification problems.
              * `f1`: usual f1-score in classification problems.
              * `accuracy`: usual accuracy in classification problems.
              * `precision_at_*`: precision at k (e.g. 'precision_at_10') or at k percent (e.g. 'precision_at_5_pct').
              * `recall_at_*`: recall at k (e.g. 'recall_at_10') or at k percent (e.g. 'recall_at_5_pct').
              * `threshold_at_*`: threshold at k (e.g. 'threshold_at_10') or at k percent (e.g. 'threshold_at_5_pct'). <br>
              Note: For the time being, all metrics have to return only one number; In the case of a multi-class classification, a micro-average precision is returned.
      </details>

    + <details>
          <summary>Using a custom function.</summary>

      If the `function` key is contained in the `evaluation_inputs` dictionary, its value must be a valid function.
      * **Arguments that this custom function must accept**: <br>
          * `execution_results` : dict <br>
              Execution results as returned by execution inputs function. It must contain a 'truth' and a 'prediction' key.
      * Additionally, this function can have an arbitrary number of optional arguments (or none), to be specified in `evaluation_inputs` dictionary.
      * **Outputs this custom function must return**: <br>
          * `results` : dict <br>
              Results of evaluation. Each element in the dictionary corresponds to one of the metrics.
      </details>

5. `exploration_inputs`: Dictionary of inputs related to the method to explore the parameter space (e.g. grid search or random search).
    + <details>
          <summary>Using the default function.</summary>

      If `function` is not given, `modev.exploration.GridSearch` will be used. <br>
      This class allows for a grid-search exploration of the parameter space.
      </details>

    + <details>
          <summary>Using a custom function.</summary>

      If the `function` key is contained in the `exploration_inputs` dictionary, its value must be a valid class.
      * **Arguments that this custom function must accept**: <br>
          * `approaches_pars` : dict <br>
              Dictionaries of approaches. Each key corresponds to one approach name, and the value is a dictionary.
              This inner dictionary of an individual approach has one key per parameter, and the value is a list of parameter values to explore.
          * `folds` : list <br>
              List of folds (e.g. `[0, 1, 2, 3]`).
          * `results` : pd.DataFrame or None <br>
              Existing results to load; None to initialise results from scratch.
      * Additionally, this class can have an arbitrary number of optional arguments (or none), to be specified in `exploration_inputs` dictionary.
      * **Methods this custom class must return**: <br>
          * `initialise_results` : function <br>
              Initialise results dataframe and return it.
          * `select_executions_left` : function <br>
              Select rows of results left to be executed and return the number of rows.
          * `get_next_point` : function <br>
              Return next point of parameter space to be explored.
      </details>
6. `selection_inputs`: Dictionary of inputs related to the model selection method.
    + <details>
          <summary>Using the default function.</summary>

      If `function` is not given, `modev.selection.model_selection` will be used. <br>
      This function takes the evaluation of approaches on some folds, and selects the best model.
      * **Arguments that must be defined in `selection_inputs`**:
          * `main_metric` : str <br>
              Name of the main metric (the one that has to be maximized).
      * **Arguments that can optionally be defined in `selection_inputs`**:
          * `aggregation_method` : str <br>
              Aggregation method to use to combine evaluations of different folds (e.g. 'mean'). <br>
              Default: 'mean'
          * `results_condition` : str or None <br>
              Condition to be applied to the results dataframe before combining results from different folds. <br>
              Default: None
          * `combined_results_condition` : str or None <br>
              Condition to be applied to the results dataframe after combining results from different folds. <br>
              Default: None
      </details>

    + <details>
          <summary>Using a custom function.</summary>

      If the `function` key is contained in the `selection_inputs` dictionary, its value must be a valid function.
      * **Arguments that this custom function must accept**: <br>
          * `results` : pd.DataFrame <br>
              Evaluations of the performance of approaches on different data folds (output of function used in `evaluation_inputs`).
      * Additionally, this function can have an arbitrary number of optional arguments (or none), to be specified in `evaluation_inputs` dictionary.
      * **Outputs this custom function must return**: <br>
          * `combine_results_sorted` : pd.DataFrame <br>
              Ranking of results (sorted in descending value of 'main_metric') of approaches that fulfil the imposed conditions.
      </details>
7. `approaches_inputs`: List of dictionaries, one per approach to be used.
    + <details>
          <summary>Definition of an approach.</summary>

      Each dictionary in the list has at least two keys:
      * `approach_name`: Name of the approach.
      * `function`: Actual approach (usually, a class with 'fit' and 'predict' methods).
      * Any other key in the dictionary of an approach will be assumed to be an argument of that approach. <br>
      To see some examples of simple approaches, see `modev.approaches.DummyPredictor` and `modev.approaches.RandomChoicePredictor`.
      </details>

An experiment can be contained in a python module.
As an example, there is a template experiment in `modev.templates`, that is a small variation with respect to the default experiment.
To start a pipeline on this experiment:
```
experiment = templates.experiment_01.experiment
pipe = Pipeline(**experiment)
```
And to run it follow the example in the quick guide.
