import itertools


def expand_grid(grid, do_not_expand=None):
    keys = []
    pars_lists = []
    if do_not_expand is None:
        do_not_expand = []
    for key in list(grid):
        value = grid[key]
        keys.append(key)
        if (type(value) == list) & (key not in do_not_expand):
            pars_lists.append(value)
        else:
            pars_lists.append([value])
    expanded_lists = list(itertools.product(*pars_lists))
    pars = []
    for comb in expanded_lists:
        odict = {keys[i]: comb[i] for i in range(len(keys))}
        pars.append(odict)
    return pars


def grid_search(grids, do_not_expand=None):
    pars_list = [expand_grid(grid, do_not_expand=do_not_expand) for grid in grids]
    return pars_list
