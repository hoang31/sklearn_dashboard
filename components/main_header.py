from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import dash_bootstrap_components as dbc

def main_header():
    content = dbc.NavbarSimple(
        brand="Interactive Sklearn Dashboard",
        brand_href="#",
        color="black",
        dark=True,
        className="mb-4",
    )
    return content