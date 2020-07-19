import importlib
import logging
import os


def get_usable_kwargs_for_function(function, kwargs):
    # Get default args for function.
    pd_read_csv_kwargs = function.__code__.co_varnames
    # Get usable keyword arguments.
    usable_kwargs = {key: kwargs[key] for key in kwargs if key in pd_read_csv_kwargs}
    # If any, warn of useless keyword arguments.
    unknown_kwargs = list(set(kwargs) - set(pd_read_csv_kwargs))
    if len(unknown_kwargs) > 0:
        logging.warning("Unknown keyword arguments %s", unknown_kwargs)
    return usable_kwargs


def import_file_as_module(file_path, module_name=None):
    """
    TODO: docstring.
    https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path

    """
    if module_name is None:
        module_name = os.path.basename(file_path).split('.')[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
