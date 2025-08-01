## =============================================
## Import packages
## =============================================


from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import plotly.express as px
from datetime import datetime
import dash_bootstrap_components as dbc


## =============================================
## DASH CODE
## =============================================


## =============================================
## Load the test data
## =============================================

df = pd.read_csv('/mnt/Data/perso/kaggle_projects/kaggle_houses_prices/data/house-prices-advanced-regression-techniques/train.csv')


def get_numerical_columns(df):
    numerical_cols = df.select_dtypes(include='number').columns.tolist()
    numerical_cols.sort()
    return numerical_cols


def get_categorical_columns(df):
    numerical_cols = df.select_dtypes(include='object').columns.tolist()
    numerical_cols.sort()
    return numerical_cols


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


## =============================================
## Initialize the app and layout
## =============================================


from components.main_header import main_header
from components.upload_data import upload_data
from components.tab__data_description import tab__data_description
from components.tab__data_modelling import tab__data_modelling
from components.tab__classification_modelling import tab__classification_modelling


## intialize the app
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])

## App layout
app.layout = dbc.Container([

    ## MAIN HEADER
    main_header(),

    ## button for date and time
    html.Br(),
    html.Button('Update', id='update-data-button', n_clicks=0),
    html.Div(id='container-button-timestamp'),
    html.Br(),

    ## UPLOAD CSV DATA
    upload_data(),
    html.Br(),
    html.Br(),

    ## TABS
    dcc.Tabs(
        id="tabs-example-graph", value='tab-1-example-graph', 
        children=[
            tab__data_description(df),
            tab__data_modelling(df),
            tab__classification_modelling(df),
        ]
    ),
    html.Div(id='tabs-content-example-graph'),

], fluid=True)


## =============================================
## CALLBACKS 
## =============================================


# Add controls to build the interaction
@callback(
    Output('container-button-timestamp', 'children'),
    Input('update-data-button', 'n_clicks'),
)
def displayClick(btn1):
    ## get the date of today
    current_datetime = datetime.now()
    current_datetime_formatted = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
    msg = f"Data Date: {current_datetime_formatted}"
    if "update-data-button" == ctx.triggered_id:
        current_datetime = datetime.now()
        current_datetime_formatted = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        msg = f"Data Date: {current_datetime_formatted}"
    return html.Div(msg)


## ========================
## Modelling
## ========================

from modules.callback_corr import callback_corr
from modules.callback_pca import callback_pca
from modules.callback_tsne import callback_kmeans
from modules.callback_kmeans import callback_kmeans

callback_corr(app, df)
callback_pca(app, df)
callback_kmeans(app, df)
callback_kmeans(app, df)


## ========================
## Display Layout dynamically
## ========================

from modules.callback_display_layout import callback_display_layout
callback_display_layout(app, df)


## =============================================
## Run the app
## =============================================


# Run the app
if __name__ == '__main__':
    app.run(debug=True)









































