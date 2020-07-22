import plotly
import plotly.express as px


def metric_vs_folds(results, main_metric, plot_file=None, added_cols_hover=None, title=None, show=True,
                    fold_col='fold', model_col='id', approach_col='approach', width=950, height=500):
    data_plot = results.copy()
    cols_hover = [model_col, approach_col]
    if added_cols_hover is not None:
        cols_hover += added_cols_hover

    fig1 = px.line(data_plot, x=fold_col, y=main_metric, width=width, height=height, hover_data=data_plot[cols_hover],
                   color=model_col)
    fig1.layout.coloraxis.showscale = False
    fig1.layout.xaxis = dict(tickvals=data_plot[fold_col].unique())
    fig1.update_traces(mode='lines+markers')
    fig1.update_layout(title=title, xaxis_title="Fold", yaxis_title=main_metric.title(), legend_title="Model ID")
    if plot_file is not None:
        plotly.offline.plot(fig1, filename=plot_file, auto_open=False)
    if show:
        fig1.show()
    return fig1
