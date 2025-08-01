

from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import plotly.express as px
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.cluster import KMeans



## ========================
## kmean
## ========================


def callback_kmeans(app, df):

    def run_kmeans(df, n, category_var, x, y):

        print('kmeans process')

        label_encoder = LabelEncoder()
        df_normalized = OneHotEncoder().fit_transform(df)
        
        df_normalized = preprocessing.normalize(df_normalized)

        ## run kmeans
        kmeans = KMeans(n_clusters = n, random_state = 0, n_init='auto')
        kmeans_res = kmeans.fit(df_normalized)
        df['kmeans_clusters'] = (kmeans_res.labels_).astype(str).tolist()

        ## generate the fig
        kmeans_fig = px.scatter(df, x=x, y=y, color='kmeans_clusters' , height=800, width=1200, template='simple_white')

        return(kmeans_fig)


    @callback(
        Output('kMEANS-plot', 'figure'),
        Input('kMEANS-input-n-values', 'value'),
        Input('kMEANS-var-to-color', 'value'),
        Input('kMEANS-x-axis', 'value'),
        Input('kMEANS-y-axis', 'value'),
        prevent_initial_call=False
    )

    def add_contents_kMMEANS(n, category_var, x, y):

        ## Run the pca
        fig_keams = run_kmeans(df, n, category_var, x, y)

        return(fig_keams)

