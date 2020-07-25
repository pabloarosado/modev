import pkg_resources

example_data_path = pkg_resources.resource_filename(__name__, 'data/example_labeled_data.csv')
random_state = 1000

etl_pars_header_nrows = 1
etl_pars_sample_nrows = None
etl_pars_selection = None

evaluation_pars_num_predictions = None

execution_pars_test_mode = False

exploration_pars_fixed_pars = None

plotting_pars_added_cols_hover = None
plotting_pars_approach_col = 'approach'
plotting_pars_fold_col = 'fold'
plotting_pars_height = 500
plotting_pars_model_col = 'id'
plotting_pars_plot_file = None
plotting_pars_show = True
plotting_pars_title = None
plotting_pars_width = 950

selection_pars_aggregation_method = 'mean'
selection_pars_condition = None
selection_pars_combined_results_condition = None
selection_pars_results_condition = None

validation_pars_labels = None
validation_pars_playground_n_folds = 4
validation_pars_return_original_indexes = True
validation_pars_shuffle = True
validation_pars_train_name = 'train'
validation_pars_test_fraction = 0.5
validation_pars_test_n_sets = 6
validation_pars_test_name = 'test'
