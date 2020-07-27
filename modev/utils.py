"""Useful functions that are not specifically designed for modev.

"""
import importlib
import logging
import os


def get_usable_args_for_function(function, args, function_args=None):
    if function_args is None:
        # Get default args for function.
        function_args = function.__code__.co_varnames
    # Get usable keyword arguments.
    usable_args = {key: args[key] for key in args if key in function_args}
    # If any, warn of useless keyword arguments.
    unknown_args = list(set(args) - set(function_args))
    if len(unknown_args) > 0:
        logging.warning("Unknown keyword arguments %s", unknown_args)
    return usable_args


def import_file_as_module(file_path, module_name=None):
    if module_name is None:
        module_name = os.path.basename(file_path).split('.')[0]
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod
