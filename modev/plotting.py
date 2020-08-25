"""Functions related to plotting.

"""
import plotly
import plotly.express as px

from modev import default_pars


approach_key = default_pars.approach_key
id_key = default_pars.id_key
fold_key = default_pars.fold_key


def metric_vs_folds(results, metric, plot_file=default_pars.plotting_pars_plot_file,
                    added_cols_hover=default_pars.plotting_pars_added_cols_hover,
                    title=default_pars.plotting_pars_title, show=default_pars.plotting_pars_show,
                    width=default_pars.plotting_pars_width, height=default_pars.plotting_pars_height):
    data_plot = results.copy()
    cols_hover = [id_key, approach_key]
    if added_cols_hover is not None:
        cols_hover += added_cols_hover

    fig1 = px.line(data_plot, x=fold_key, y=metric, width=width, height=height, hover_data=data_plot[cols_hover],
                   color=id_key)
    fig1.layout.coloraxis.showscale = False
    fig1.layout.xaxis = dict(tickvals=data_plot[fold_key].unique())
    fig1.update_traces(mode='lines+markers')
    fig1.update_layout(title=title, xaxis_title="Fold", yaxis_title=metric.title(), legend_title="Model ID")
    if plot_file is not None:
        plotly.offline.plot(fig1, filename=plot_file, auto_open=False)
    if show:
        fig1.show()
    return fig1
