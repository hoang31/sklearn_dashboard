
from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import dash_bootstrap_components as dbc


def tab__classification_modelling(df):
    tab_content = dcc.Tab(
        label='Classification Modelling',
        value='tab-classification-modelling',
        children=[
            html.Div([html.Div(style={'height': '50px'})]),
            #html.Hr(),
            html.Br(),


            html.Div(
                [
                    dbc.Card(
                        [
                            html.P(""),
                            html.Div([
                                dbc.Row([
                                    dbc.Col(html.H4("Split Data to Train/Test Set"), width=3),
                                    dbc.Col(
                                        dcc.Input(
                                            id = 'split-data-input',
                                            placeholder='Training Set Proportion',
                                            type='text',
                                            value='0.8'
                                        ),
                                        width=1
                                    ),
                                    dbc.Col(width=10),
                                ]),
                            ]),
                            html.P(""),

                        ],
                        className="mb-3",
                    ),
                    dbc.Card(
                        [
                            html.P(""),
                            dcc.Checklist(
                                ['Cross Validation', 'Stratified Cross Validation', ' Leave-One-Out Cross-Validation'],
                                ['Cross Validation'],
                                id='cross-validation-checklist',
                                inline=False
                            ),
                            html.P(""),
                        ],
                        className="mb-3",
                    ),
                    #dbc.Card("This is also within a body", body=True),
                ]
            ),
            html.Br(),
            html.Div([
                dbc.Row([
                    dbc.Col(html.H4("Training Set",  style={'textAlign': 'center'}), width=6),
                    dbc.Col(html.H4("Testing Set",  style={'textAlign': 'center'}), width=6),
                ]),
            ]),
        ]
    )
    return(tab_content)