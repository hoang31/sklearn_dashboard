# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import plotly.express as px
from datetime import datetime
import dash_bootstrap_components as dbc


from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder




# Incorporate data
df = pd.read_csv('/mnt/Data/perso/kaggle_projects/kaggle_houses_prices/data/house-prices-advanced-regression-techniques/train.csv')
#df = pd.read_csv('/mnt/Data/perso/kaggle_projects/kaggle__World_Bank_Dataset/workflow/data/world_bank_dataset.csv')









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



from components.tab__data_description import tab__data_description
from components.tab__data_modelling import tab__data_modelling
from components.tab__classification_modelling import tab__classification_modelling



# Initialize the app
#app = Dash()
app = Dash(external_stylesheets=[dbc.themes.BOOTSTRAP])
#app = Dash(external_stylesheets=[dbc.themes.CYBORG])

# App layout
app.layout = dbc.Container([

    dbc.NavbarSimple(
        brand="Interactive Sklearn Dashboard",
        brand_href="#",
        color="black",
        dark=True,
        className="mb-4",
    ),

    ## button for date and time
    html.Br(),
    html.Button('Update', id='update-data-button', n_clicks=0),
    html.Div(id='container-button-timestamp'),
    html.Br(),

    html.Div(
        dbc.Accordion(
            [
                dbc.AccordionItem(
                    [
                        #html.H3(html.B("Upload your csv file here")),
                        dcc.Upload(
                            id='upload-data',
                            children=html.Div([
                                'Drag and Drop or ',
                                html.A('Select Files')
                            ]),
                            style={
                                'width': '100%',
                                'height': '60px',
                                'lineHeight': '60px',
                                'borderWidth': '1px',
                                'borderStyle': 'dashed',
                                'borderRadius': '5px',
                                'textAlign': 'center',
                                'margin': '10px'
                            },
                            # Allow multiple files to be uploaded
                            multiple=False
                        ),
                    ],
                    title=html.P(html.B("Click Here to Upload CSV Data")),
                ),
                
            ],
            start_collapsed=True
        )
    ),

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
## Corr
## ========================

from modules.callback_corr import callback_corr
callback_corr(app, df)


## ========================
## PCA
## ========================

from modules.callback_pca import callback_pca
callback_pca(app, df)


### ========================
### Tsne
### ========================

from modules.callback_tsne import callback_kmeans
callback_kmeans(app, df)


## ========================
## kmean
## ========================

from modules.callback_kmeans import callback_kmeans
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









































