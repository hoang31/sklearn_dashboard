

from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import dash_bootstrap_components as dbc



def table_metrics_numerical_columns(df):

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



def tab__data_description(df):
    tab_content = dcc.Tab(
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

        ]
    )

    return(tab_content)