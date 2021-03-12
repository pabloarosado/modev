"""Default values for modev parameters.

"""
import pkg_resources

approach_key = 'approach'
approach_name_key = 'approach_name'
dev_key = 'dev'
example_data_path = pkg_resources.resource_filename(__name__, 'data/example_labeled_data.csv')
executed_key = 'executed'
fixed_pars_key = 'fixed_pars'
fold_key = 'fold'
function_key = 'function'
id_key = 'id'
pars_key = 'pars'
playground_key = 'playground'
prediction_key = 'prediction'
random_state = 1000
save_every = 10
test_key = 'test'
train_key = 'train'
truth_key = 'truth'

etl_pars_header_nrows = 1
etl_pars_sample_nrows = None
etl_pars_selection = None

evaluation_pars_num_predictions = None

exploration_pars_fixed_pars = None

plotting_pars_added_cols_hover = None
plotting_pars_height = 500
plotting_pars_plot_file = None
plotting_pars_show = True
plotting_pars_title = None
plotting_pars_width = 950

selection_pars_aggregation_method = 'mean'
selection_pars_condition = None
selection_pars_combined_results_condition = None
selection_pars_results_condition = None

validation_pars_test_fraction = 0.5
validation_pars_test_n_sets = 6
# Default validation pars for k-fold cross-validation:
validation_pars_labels = None
validation_pars_playground_n_folds = 4
validation_pars_return_original_indexes = True
validation_pars_shuffle = True
validation_pars_test_mode = False
# Default validation pars for temporal-fold cross-validation:
validation_min_n_train_examples = 10
validation_dev_n_sets = 4
