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
We define an **experiment** as a combination of the inputs that a pipeline accepts:
1. ```load_inputs```: Dictionary of inputs related to data loading.
    To read what the default function does, see documentation of ```modev.etl.load_local_file```.
2. ```validation_inputs```: Dictionary of inputs related to validation method (e.g. k-fold or temporal-fold
    cross-validation).
    To read what the default function does, see documentation of ```modev.validation.k_fold_playground_n_tests_split```.
3. ```execution_inputs```: Dictionary of inputs related to the execution of approaches (by default, an approach consists
    of a class with a 'fit' and a 'predict' method).
    To read what the default function does, see documentation of ```modev.execution.execute_model```.
4. ```evaluation_inputs```: Dictionary of inputs related to evaluation metrics.
    To read what the default function does, see documentation of ```modev.evaluation.evaluate_predictions```.
5. ```exploration_inputs```: Dictionary of inputs related to the method to explore the parameter space (e.g. grid search
    or random search).
    To read what the default function does (in fact, in this case it is a class, not a function), see documentation of
    ```modev.exploration.GridSearch```.
6. ```selection_inputs```: Dictionary of inputs related to the model selection method.
    To read what the default function does, see documentation of ```modev.selection.model_selection```.
7. ```approaches_inputs```: List of dictionaries, one per approach to be used.
    To read what the default function does, see documentation of default approaches
    (```modev.approaches.DummyPredictor``` and ```modev.approaches.RandomChoicePredictor```).

If any of these inputs is not given, they will be taken from default.
Each dictionary has a key ```'function'``` that corresponds to a particular function (or class).
Any other item in the dictionary is assumed to be an argument of that function.
* If ```'function'``` is not specified, a default function (taken from one of modev's modules) is used.
Arguments of the function can be given in the same dictionary (and if not given, default values are assumed).
* If a custom ```'function'``` is specified, all parameters required by it can be given in the same dictionary.

There is one special case: ```approaches_inputs``` is not a dictionary but a list of dictionaries (given that one
experiment can include several approaches).
Each dictionary in the list has at least two keys:
* ```approach_name```: Name of the approach.
* ```function```: Actual approach (usually, a class with 'fit' and 'predict' methods).
* Any other given key will be assumed to be arguments of the approach.

An experiment can be contained in a python module.
As an example, there is a template experiment in ```modev.templates```, that is a small variation with respect to the
default experiment.
To start a pipeline on this experiment:
```
experiment = templates.experiment_01.experiment
pipe = Pipeline(**experiment)
```
And to run it follow the example in the quick guide.
