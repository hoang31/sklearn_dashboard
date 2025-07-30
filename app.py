# Import packages
from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import plotly.express as px
from datetime import datetime
import dash_bootstrap_components as dbc
import plotly.io as pio


# Incorporate data
df = pd.read_csv('/mnt/Data/perso/kaggle_projects/kaggle_houses_prices/data/house-prices-advanced-regression-techniques/train.csv')
#df = pd.read_csv('/mnt/Data/perso/kaggle_projects/kaggle__World_Bank_Dataset/workflow/data/world_bank_dataset.csv')



## functions

def table_metrics_numerical_columns(df):

    print("okkkkkkkkkkkkkkkkkkk")

    data_info_numeric = df.describe().T.reset_index()
    data_info_numeric = data_info_numeric.rename(columns={'index' : 'column'})
    data_info_numeric = data_info_numeric.sort_values(by=['column'])
    
    ## generate the table to display
    data_info_numeric_displayed = html.Div([
        dash_table.DataTable(
            data_info_numeric.to_dict('records'),
            [{"name": i, "id": i} for i in data_info_numeric.columns],
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            selected_columns=[],
            style_header={
                'textAlign': 'center',
                'backgroundColor': 'rgb(210, 210, 210)',
                'color': 'black',
                'fontWeight': 'bold'
            },
            style_table={
                        'width': 'auto',
                        'textAlign': 'center',
                        'overflowX': 'auto'
                        },
            style_cell={'textAlign': 'left'},
            style_data={
                        'color': 'black',
                        'backgroundColor': 'white'
                        },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(220, 220, 220)',
                }
            ],
            fill_width=True
        )
    ],
    style={'marginLeft': 'auto', 'marginRight': 'auto', 'width': '100%', 'display': 'block'}
    )
    return data_info_numeric_displayed




def table_metrics_categorical_columns(df):

    ## initialize the list containing the data
    category_data_count = list()

    ## get the categorical data
    data_categorical = df.select_dtypes(include='object')

    ## count the number of categories
    cols_category = data_categorical.columns.tolist()
    cols_category.sort()

    for col in cols_category:
        dt_counting = data_categorical[col].value_counts().reset_index()
        dt_counting['count_string'] = dt_counting.apply(lambda x: x[col] + ": " + str(x["count"]), axis=1)

        ## create dic
        category_data_count.append("\n".join(dt_counting['count_string'].tolist()))

    ## generate the dt
    data = {'columns': cols_category, 'counts': category_data_count}
    data = pd.DataFrame(data)


    data_displayed = html.Div([
        dash_table.DataTable(
            data.to_dict('records'),
            [{"name": i, "id": i} for i in data.columns],
            filter_action="native",
            sort_action="native",
            sort_mode="multi",
            column_selectable="single",
            selected_columns=[],
            style_header={
                'textAlign': 'center',
                'backgroundColor': 'rgb(210, 210, 210)',
                'color': 'black',
                'fontWeight': 'bold'
            },
            style_table={
                        'width': 'auto',
                        'textAlign': 'center',
                        'overflowX': 'auto'
                        },
            style_cell={
                'textAlign': 'left',
                'whiteSpace': 'pre-line'
            },
            style_data={
                        'color': 'black',
                        'backgroundColor': 'white'
                        },
            style_data_conditional=[
                {
                    'if': {'row_index': 'odd'},
                    'backgroundColor': 'rgb(220, 220, 220)',
                }
            ],
            fill_width=True
        )
    ], 
    #style={'marginLeft': 'auto', 'marginRight': 'auto', 'width': '50%', 'display': 'block'}
    )

    return data_displayed


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


print("=======================")
get_numerical_columns(df)





































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



    #html.Div(
    #    [
    #        #html.Div(style={'height': '50px'}),
    #        html.H3(html.B("Upload Data"), style={'textAlign': 'center'}),
    #    ],
    #    id = 'upload-data-header'
    #),


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



    dcc.Tabs(
        id="tabs-example-graph", value='tab-1-example-graph', 
        children=[

            ### tabs 1
            #dcc.Tab(
            #    label='Data Description',
            #    value='tab-data-description',
            #    children=[
            #        html.Br(),
            #        html.Br(),
            #        dbc.Row(
            #            [
            #                dbc.Col(html.H2("Data Information - Numerical"), width=4),
            #                dbc.Col(html.H2("Data Information - Categorical"), width=8),
            #            ],
            #        ),
            #        dbc.Row(
            #            [
            #                dbc.Col(table_metrics_numerical_columns(df), width=4),
            #                dbc.Col(width=1),
            #                dbc.Col(table_metrics_categorical_columns(df), width=7),
            #            ],
            #        ),
            #            html.Br()
            #    ]
            #),


            dcc.Tab(
                label='Data Description',
                value='tab-data-description',
                children=[
                    html.Br(),
                    html.Br(),

                    html.Div(
                        dbc.Accordion(
                            [
                                dbc.AccordionItem(
                                    [
                                        #html.P("This is the content of the first section"),
                                        html.H2(html.B("Data Information - Numerical"), style={'textAlign': 'center'}),
                                        table_metrics_numerical_columns(df),
                                    ],
                                    title="Description - Numerical",
                                ),
                                dbc.AccordionItem(
                                    [
                                        html.H2(html.B("Data Information - Categorical"), style={'textAlign': 'center'}),
                                        table_metrics_categorical_columns(df),
                                    ],
                                    title="Description - Categorical",
                                ),
                            ],
                            start_collapsed=True,
                        )
                    ),

                    #dbc.Row(
                    #    [
                    #        html.H2(html.B("Data Information - Numerical"), style={'textAlign': 'center'}),
                    #        table_metrics_numerical_columns(df),
                    #    ],
                    #),
                    #html.Br(),
                    #html.Br(),
                    #dbc.Row(
                    #    [
                    #        html.H2(html.B("Data Information - Categorical"), style={'textAlign': 'center'}),
                    #        table_metrics_categorical_columns(df),
                    #    ],
                    #),
                ]
            ),



            ## tabs 3 
            dcc.Tab(
                label='Data Modelling',
                value='tab-data-modelling',
                children=
                    [
                        html.Br(),
                        html.Br(),
                        html.H3("Choose Model"),
                        dcc.Dropdown(
                            get_models(),
                            None,
                            multi=False,
                            id = "modelling-dropdown-model"
                        ),
                        html.Br(),
                        html.Br(),
                        
                        html.Hr(),

                        ###############################################
                        ## Correlation Analysis
                        ###############################################

                        html.Div(
                            [
                                html.H3(html.B("Correlation Analysis")),
                                html.Div(style={'height': '50px'}),
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
                                    get_numerical_columns(df)[0],
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
                                html.H3(html.B("Principal Component Analysis")),
                                html.Div(style={'height': '50px'}),
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
                                html.H3(html.B("t-SNE")),
                                html.Div(style={'height': '50px'}),
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
                                html.H3(html.B("k-Means Clustering")),
                                html.Div(style={'height': '50px'}),
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
                        html.Div([html.Div(style={'height': '50px'})], id = 'kMEANS-spacer1'),

                        html.H4("k-Means Visualization", id='kMEANS-plot-header', style = {'display': 'block'}),
                        html.Div([dcc.Graph(id='kMEANS-plot')], style= {'display': 'block'}),
                    ]
            ),
            
            
            
            
            
            
            
            
            dcc.Tab(
                label='Classification Modelling',
                value='tab-classification-modelling',
                children=[
                    html.Div([html.Div(style={'height': '50px'})]),
                    html.Hr(),
                    html.Br(),

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

                    html.Br(),
                    html.Hr(),
                    html.Br(),


                    dcc.Checklist(
                        ['Cross Validation', 'Stratified Cross Validation', ' Leave-One-Out Cross-Validation'],
                        ['Cross Validation'],
                        id='cross-validation-checklist',
                        inline=False
                    ),

                    html.Br(),
                    html.Hr(),
                    html.Br(),

                    html.Div([
                        dbc.Row([
                            dbc.Col(html.H4("Training Set",  style={'textAlign': 'center'}), width=6),
                            dbc.Col(html.H4("Testing Set",  style={'textAlign': 'center'}), width=6),

                        ]),

                    ]),






                ]
            ),





        ]),
    html.Div(id='tabs-content-example-graph'),

], fluid=True)



























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


## ========================
## PCA
## ========================


def run_impute_nan(df):
    df = df.select_dtypes(include='float')
    df = df.fillna(df.mean())
    return df   

def run_pca(df, n_pc, category_var):
    
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

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


## ========================
## Tsne
## ========================


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























## ========================
## Display Layout dynamically
## ========================


def get_all_ids_from_layout(component):
    """
    Recursively extracts all component IDs from a Dash layout - related to the models list
    """
    ids = []
    if hasattr(component, 'id') and component.id is not None:
        ids.append(component.id)

    if hasattr(component, 'children') and component.children is not None:
        if isinstance(component.children, list):
            for child in component.children:
                ids.extend(get_all_ids_from_layout(child))
        elif hasattr(component.children, 'id'):  # Single child component
            ids.extend(get_all_ids_from_layout(component.children))

    ## current models list
    models_list = get_models()
    models_list = [model_i.replace('-', "") for model_i in models_list]

    ## filter the ids based on the model list
    ids = [id_i for id_i in ids for model_i in models_list if model_i in id_i]
    return ids

all_ids = get_all_ids_from_layout(app.layout)


## create dict to dynimically display the layout
def display_layout(output_id_dt, model_name_input):

    ## creating the dt containing all the output to display
    output_id_dt = pd.DataFrame(output_id_dt, columns=['output_id'])
    output_id_dt['display_value'] = "none"

    ## change the display value to block for the output_id_dt related to the model_name_input
    if not model_name_input == None:
        output_id_dt.loc[output_id_dt['output_id'].str.contains(model_name_input), 'display_value'] = 'block'

    print("=====")
    print(model_name_input)
    print(output_id_dt)

    ## initialize the dict and sublist
    list_bool_to_display = []

    for i in range(output_id_dt.shape[0]):
        list_bool_to_display.append({'display': output_id_dt.iloc[i]['display_value']})

    return(list_bool_to_display)

@callback(
    [Output(component_id=id_i, component_property='style') for id_i in all_ids],
    Input('modelling-dropdown-model', 'value'),
    prevent_initial_call=False
)

def add_general_layout(model_name):

    if not model_name == None:
        model_name_formatted = model_name.replace('-', "")
        model_name_formatted = model_name_formatted + '-'
    else:
        model_name_formatted = None
        
    ## get the output id from the callback context
    output_id_list = ctx.outputs_list
    output_id_list = [id_output['id'] for id_output in output_id_list]

    ## create the list of content to display
    content_to_display = display_layout(output_id_list, model_name_formatted)

    return(content_to_display)


























# Run the app
if __name__ == '__main__':
    app.run(debug=True)









































