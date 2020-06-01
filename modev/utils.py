import logging


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
