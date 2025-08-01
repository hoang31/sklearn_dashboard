from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import dash_bootstrap_components as dbc

def upload_data():
    content = html.Div(
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
    )
    return content