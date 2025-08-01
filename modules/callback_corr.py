

from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import plotly.express as px


def callback_corr(app, df):
        
    @callback(
        Output('corr-graph-corr-plot', 'figure'),
        Input('corr-x-axis', 'value'),
        Input('corr-y-axis', 'value'),
        Input('corr-var-to-color', 'value'),
        prevent_initial_call=False
    )
    def generate_correlation_plot(x, y, category_var):
        print('Corr Process')

        if category_var != None:
            fig = px.scatter(df, x=x, y=y, height=800, width=1200, template='simple_white', color=df[category_var])
        else:
            fig = px.scatter(df, x=x, y=y, height=800, width=1200, template='simple_white')

        fig.update_layout(font=dict(size=18))
        fig.update_traces(marker=dict(size=10))
        return fig