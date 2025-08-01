

from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import plotly.express as px
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing



def callback_kmeans(app, df):

    def run_impute_nan(df):
        df = df.select_dtypes(include='float')
        df = df.fillna(df.mean())
        return df   


    def run_tsne(df, n, category_var):

        from sklearn.manifold import TSNE

        print('TSNE process')
        df_filtered = df.select_dtypes(include=['float64', 'int64'])
        df_filtered = run_impute_nan(df_filtered)

        ## run the tsne
        tsne = TSNE(n_components=n, random_state=42)
        tsne_res = tsne.fit_transform(df_filtered)
        tsne_res = pd.DataFrame(tsne_res)

        #print(tsne_res[:, 0])
        ## generate the fig
        tsne_fig = px.scatter(x=tsne_res.loc[:,0], y=tsne_res.loc[:,1], color=df[category_var], height=800, width=1200, template='simple_white')
        
        return(tsne_fig)

    @callback(
        Output('tSNE-plot', 'figure'),
        Input('tSNE-input-n-values', 'value'),
        Input('tSNE-var-to-color', 'value'),
        prevent_initial_call=False
    )

    def add_contents_tsne(n, category_var):

        ## Run the pca
        fig_tse = run_tsne(df, n, category_var)

        return(fig_tse)
