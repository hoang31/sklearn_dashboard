
from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import dash_bootstrap_components as dbc



def get_models():
    models = [
        'corr',
        'PCA',
        't-SNE',
        'k-MEANS',
        #"Logistic Regression",
        #"Random Forest"
    ]
    return models

def get_numerical_columns(df):
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    numerical_cols.sort()
    return numerical_cols

def get_categorical_columns(df):
    numerical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols.sort()
    return numerical_cols

def tab__data_modelling(df):
    tab_content = dcc.Tab(
        label='Data Modelling',
        value='tab-data-modelling',
        children=
            [
                html.Br(),
                html.Br(),
                html.Br(),

                dbc.Card(
                    [
                        html.P(""),
                        html.H3("Choose Model", style={'textAlign': 'center'}),
                        dcc.Dropdown(
                            get_models(),
                            None,
                            multi=False,
                            id = "modelling-dropdown-model"
                        ),
                        html.P(""),
                    ]
                ),
                html.Br(),
                html.Br(),                        
                html.Br(),                        
                html.Br(),                        


                ###############################################
                ## Correlation Analysis
                ###############################################


                html.Div(
                    [
                        html.H3(html.B("Correlation Analysis"), style = {'textAlign': 'center'}),
                        html.Div(style={'height': '30px'}),
                    ],
                    id = 'corr-header'
                ),


                html.H4("X Axis", id='corr-x-axis-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Dropdown(
                            get_numerical_columns(df),
                            get_numerical_columns(df)[0],
                            multi=False,
                            id = "corr-x-axis"
                        ),
                    ],
                    style= {'display': 'block'}
                ),
                html.H4("y Axis", id='corr-y-axis-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Dropdown(
                            get_numerical_columns(df),
                            get_numerical_columns(df)[1],
                            multi=False,
                            id = "corr-y-axis"
                        ),
                    ],
                    style= {'display': 'block'}
                ),
                html.H4("Color by", id='corr-var-to-color-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Dropdown(
                            get_categorical_columns(df),
                            None,
                            multi=False,
                            id = "corr-var-to-color"
                        ),
                    ],
                    style= {'display': 'block'}
                ),
                html.Div([html.Div(style={'height': '20px'})],
                    id = 'corr-spacer1'
                ),
                html.Div(
                    [
                        dcc.Graph(id='corr-graph-corr-plot')
                    ],
                    style= {'display': 'block'}
                ),


                ###############################################
                ## PCA
                ###############################################

                html.Div(
                    [
                        html.H3(html.B("Principal Component Analysis"), style = {'textAlign': 'center'}),
                        html.Div(style={'height': '30px'}),
                    ],
                    id = 'PCA-header'
                ),
    
                html.H4("Number of Components", id='PCA-input-n-values-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Input(
                            id='PCA-input-n-values',
                            placeholder='Principal Components N...',
                            type='number',
                            value=2
                        )
                    ],
                    style= {'display': 'block'}
                ),
                html.Div(id='PCA-pc-values', style= {'display': 'block'}),
                html.Br(id='PCA-spacing-1', style= {'display': 'block'}),
                html.Br(id='PCA-spacing-2', style= {'display': 'block'}),
                html.H4("Color by", id='PCA-var-to-color-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Dropdown(
                            get_categorical_columns(df),
                            get_categorical_columns(df)[0],
                            multi=False,
                            id = "PCA-var-to-color"
                        ),
                    ],
                    style= {'display': 'block'}
                ),
                html.Div([html.Div(style={'height': '50px'})], id = 'PCA-spacer1'),
                html.H4("PCA Visualization", id='PCA-plot-header', style = {'display': 'block'}),
                html.Div([dcc.Graph(id='PCA-plot')], style= {'display': 'block'}),
                html.H4("PCA Visualization - Variable Contribution", id='PCA-plot-var-contribution-header', style = {'display': 'block'}),
                html.Div([dcc.Graph(id='PCA-plot-var-contribution')], style= {'display': 'block'}),



                ###############################################
                ## T-SNE
                ###############################################

                html.Div(
                    [
                        html.H3(html.B("t-SNE"), style = {'textAlign': 'center'}),
                        html.Div(style={'height': '30px'}),
                    ],
                    id = 'tSNE-header'
                ),
                html.H4("Number of t-SNE Components", id='tSNE-input-n-values-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Input(
                            id='tSNE-input-n-values',
                            placeholder='TSNE Components N...',
                            type='number',
                            value=2
                        )
                    ],
                    style= {'display': 'block'}
                ),
                html.Div(id='tSNE-pc-values', style= {'display': 'block'}),
                html.H4("Color by", id='tSNE-var-to-color-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Dropdown(
                            get_categorical_columns(df),
                            get_categorical_columns(df)[0],
                            multi=False,
                            id = "tSNE-var-to-color"
                        ),
                    ],
                    style= {'display': 'block'}
                ),
                html.Div([html.Div(style={'height': '50px'})], id = 'tSNE-spacer1'),
                html.H4("t-SNE Visualization", id='tSNE-plot-header', style = {'display': 'block'}),
                html.Div([dcc.Graph(id='tSNE-plot')], style= {'display': 'block'}),


                ###############################################
                ## k-MEANS
                ###############################################

                html.Div(
                    [
                        html.H3(html.B("k-Means Clustering"), style = {'textAlign': 'center'}),
                        html.Div(style={'height': '30px'}),
                    ],
                    id = 'kMEANS-header'
                ),
                html.H4("Number of kMEANS", id='kMEANS-input-n-values-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Input(
                            id='kMEANS-input-n-values',
                            placeholder='N of K-MEANS...',
                            type='number',
                            value=2
                        )
                    ],
                    style= {'display': 'block'}
                ),
                html.Div(id='kMEANS-k-value', style= {'display': 'block'}),
                html.H4("Color by", id='kMEANS-var-to-color-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Dropdown(
                            ["k-Means-Clusters"] + get_categorical_columns(df),
                            "k-Means-Clusters",
                            multi=False,
                            id = "kMEANS-var-to-color"
                        ),
                    ],
                    style= {'display': 'block'}
                ),

                html.H4("X Axis", id='kMEANS-x-axis-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Dropdown(
                            get_numerical_columns(df),
                            get_numerical_columns(df)[0],
                            multi=False,
                            id = "kMEANS-x-axis"
                        ),
                    ],
                    style= {'display': 'block'}
                ),
                html.H4("y Axis", id='kMEANS-y-axis-header', style= {'display': 'block'}),
                html.Div(
                    [   
                        dcc.Dropdown(
                            get_numerical_columns(df),
                            get_numerical_columns(df)[1],
                            multi=False,
                            id = "kMEANS-y-axis"
                        ),
                    ],
                    style= {'display': 'block'}
                ),


                html.Div([html.Div(style={'height': '50px'})], id = 'kMEANS-spacer1'),
                html.H4("k-Means Visualization", id='kMEANS-plot-header', style = {'display': 'block'}),
                html.Div([dcc.Graph(id='kMEANS-plot')], style= {'display': 'block'}),
            ]
    )

    return(tab_content)