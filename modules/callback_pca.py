

from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import plotly.express as px
from datetime import datetime

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def callback_pca(app, df):

    def run_impute_nan(df):
        df = df.select_dtypes(include='float')
        df = df.fillna(df.mean())
        return df   

    def run_pca(df, n_pc, category_var):
        


        print('PCA process')

        df_filtered = df.select_dtypes(include=['float64', 'int64'])
        df_filtered = run_impute_nan(df_filtered)

        ## scale the data
        scaler = StandardScaler()
        df_filtered_scaled = scaler.fit_transform(df_filtered)

        ## run PCA
        pca = PCA(n_components=n_pc)
        res_pca = pca.fit_transform(df_filtered_scaled)

        ## pca variable contribution
        pca_variable_contribution = pd.DataFrame(pca.components_, columns=df_filtered.columns).T
        pca_variable_contribution.columns = [f'PCA {col_i + 1}' for col_i in range(len(pca_variable_contribution.columns))]

        ## np to dataframe
        res_pca = pd.DataFrame(res_pca)

        ## generate the labels
        labels = {
            str(i): f"PC {i+1} ({var:.1f}%)"
            for i, var in enumerate(pca.explained_variance_ratio_ * 100)
        }

        ## adjust the wdth based on the number of components
        fig_width = 1000 + n_pc * 150
        fig_height = 700 + n_pc * 75

        ## generate the plot
        fig_pca = px.scatter_matrix(
            res_pca,
            dimensions=range(n_pc),
            labels=labels,
            color=df[category_var],
            height=fig_height,
            width=fig_width,
            template='simple_white'
        )

        ## adjust the wdth based on the number of components
        fig_width = 1000 + n_pc * 150
        fig_height = 700

        ## generate the plot of variable contribution
        fig_pca_var_contribution = px.imshow(
            pca_variable_contribution,
            height=fig_height,
            width=fig_width,
            color_continuous_scale='RdBu',
            zmin=-1, zmax=1,
            template='simple_white',
            text_auto=True
        )

        return res_pca, pca, pca_variable_contribution, fig_pca, fig_pca_var_contribution

    @callback(
        Output('PCA-plot', 'figure'),
        Output('PCA-plot-var-contribution', 'figure'),
        Input('PCA-input-n-values', 'value'),
        Input('PCA-var-to-color', 'value'),
        prevent_initial_call=False
    )

    def add_contents_pca(n,category_var):

        ## Run the pca
        pca_res, pca_object, pca_variable_contribution, fig_pca, fig_pca_var_contribution = run_pca(df, n, category_var)

        return(fig_pca, fig_pca_var_contribution)