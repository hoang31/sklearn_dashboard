

from dash import Dash, html, dash_table, dcc, callback, Output, Input, ctx, callback_context
import pandas as pd
import plotly.express as px



def callback_display_layout(app, df):
    
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


