#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
#general plotly dependencies
import plotly.plotly as py
import plotly.offline as offline
import plotly.graph_objs as go
import numpy as np

#enables plotly to work offline
from plotly import tools
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
init_notebook_mode()
import warnings
warnings.filterwarnings("ignore")

import json


# In[2]:


df = pd.read_csv('all_images_predictions.csv')

#remove unwanted images
df = df[~df['filename'].isin(['output.jpg', 'test.jpg'])]

def replace_spaces_filenames(data, reference_point, distance):
    """
    Ensures that all filenames have the same formatting

    """
    for i in range(len(data)):
        space_index = data.loc[i, 'filename'].index(reference_point) + distance
        if data.loc[i, 'filename'][space_index] == '_':
            continue
        elif data.loc[i, 'filename'][space_index] == ' ':
            data.loc[i, 'filename'] = data.loc[i, 'filename'].replace(' ', '_')
        else:
            data.loc[i, 'filename'] = data.loc[i, 'filename'][:space_index] + ' ' + data.loc[i, 'filename'][space_index:]
    return data

df = replace_spaces_filenames(df, '2018', 10)
df['filename'] = df.filename.str.replace(' ', '_')

def filename_to_datetime(data, date_index, time_index, string_format):
    """
    extracts date and time from filenames and converts it to datetime objects
    """
    data['date'] = data.filename.str.split('_').str.get(date_index)
    data['time'] = data.filename.str.split('_').str.get(time_index)
    data['time'] = data['time'].str[:-3]
    data['datetime'] = data['date'] + ' ' + data['time']
    data['datetime'] = pd.to_datetime(data['datetime'], format = string_format)
    data['date-hour'] = [x.replace(minute = 0) for x in data['datetime']]
    
    return data

df = filename_to_datetime(df, 3, 4, "%Y-%m-%d %H-%M-%S")


# In[3]:


df['river'] = df['filename'].str.split('_').str.get(0)
df['bridge_location'] = df.filename.str.split('_').str.get(1) + ' ' + df.filename.str.split('_').str.get(2)
df['river+location'] = df['river'] + '-' + df['bridge_location']
df['count'] = 1


# In[5]:


area_dict = {
    'Grogol' : 7.77143 * 5.8757, 
    'Kaliadem' : 9.4286 * 7.0714,
    'Marunda' : 11.1429 * 8.3571,
    'Cengkareng' : 6.8571 * 5.1429,
    'Ciliwung' : 13.7143 * 10.2857
}


# In[4]:


def image_count_to_meter(data, area_dict, bridge_ratio):
    """
    Input:
    data: grouped dataset
    count: name of the count variable
    area_dict: dictionary containing the area estimates of the images for each location
    bridge_ratio: how much of the image contains bridge rather than river
    
    Output:
    Grouped dataset with a plastic count per meter included
    """
    data['monitored_area'] = data['river'].map(area_dict)
    data['meter_count'] = data['count'] / (data['monitored_area'] * bridge_ratio)
    
    return data


# In[6]:


def dataframe_count_formatting(data, confidence_threshold, groupby_variables, sort_variables, select_variables): 
    data['meter_count'] = 1 #initiate count variable
    
    #select only observations with more confidence then the threshold
    data_theshold = data[data['score'] >= confidence_threshold]

    #group data based on datetime and river and sum the count the rest
    df_grouped = data_theshold.groupby(groupby_variables).count().reset_index()
    
    df_grouped = image_count_to_meter(df_grouped, area_dict, 0.9)

    #sort dataframe based on river and datetime
    df_sort = df_grouped.sort_values(by = sort_variables)
    df_sort = df_sort.reset_index()

    #select only desired columns
    df_select = df_sort[select_variables]
    
    df_select['meter_count_str'] = df_select['meter_count'].astype('str')
    
    return df_select

groupby_variables = ['datetime', 'time', 'date-hour', 'river']
sort_variables = ['river', 'datetime']
select_variables = ['datetime', 'time',  'date-hour', 'count', 'meter_count', 'river']

df_select = dataframe_count_formatting(df, 0.309, groupby_variables, sort_variables, select_variables)


groupby_variables = ['datetime', 'time', 'date-hour', 'river', 'river+location']
sort_variables = ['river', 'river+location', 'datetime']
select_variables = ['datetime', 'time',  'date-hour', 'count', 'meter_count', 'river', 'river+location']

df_select2 = dataframe_count_formatting(df, 0.309, groupby_variables, sort_variables, select_variables)


# In[8]:


def extract_session_id(data, time_difference_threshold, column):
    """
    This function takes a dataframe with datime, count and river variables, 
    and returns the dataframe with a session ID added depending on gaps between observations,
    and a list containing the sessions that are the first of new locations
    """
    data['session'] = 1
    session = 1
    new_location_session = [1]
    for i in range(1, len(data)):
        seconds_difference = (data.loc[i, 'datetime'] - data.loc[i-1, 'datetime']).seconds
        data.loc[i, 'session'] = session
        if seconds_difference > time_difference_threshold:
            session += 1
            data.loc[i, 'session'] = session
        elif data.loc[i, column] != data.loc[i-1, column]:
            session += 1
            data.loc[i, 'session'] = session
        if data.loc[i, column] != data.loc[i-1, column]:
            new_location_session.append(session)
            
    data['session_str'] = data['session'].astype('str')
    session_list = list(set(data.session))
            
    return data, new_location_session, session_list

df_select, new_location_session, session_list = extract_session_id(df_select, 10800, 'river')

df_select2, new_location_session2, session_list2 = extract_session_id(df_select2, 10800, 'river')


# In[11]:


#counts the number of objects per river
df_river_count = df.groupby(['filename', 'river']).count().reset_index()
df_river_count = image_count_to_meter(df_river_count, area_dict, 0.9)
df_river_count = df_river_count[['river', 'count', 'meter_count', 'filename']]

#counts the number of images per river
df_river_count['image_count'] = 1
df_river_count = df_river_count.groupby(['river']).mean().reset_index()
df_river_count = df_river_count[['river', 'meter_count', 'image_count']]

#the order that it should be, this is just easy for now, can be avoided in the future
river_list = ['Grogol', 'Cengkareng', 'Ciliwung', 'Kaliadem', 'Marunda']
df_river_count['river_cat'] = pd.Categorical(df_river_count['river'], categories = river_list, ordered = True)
df_river_count = df_river_count.sort_values('river_cat')
del df_river_count['river_cat']
df_river_count['latitude'] = [-6.1532229, -6.1433984, -6.1255507, -6.1166527, -6.1053077]
df_river_count['longitude'] = [106.7944087, 106.749266, 106.8292338, 106.7729798, 106.9686978]

df_river_count['objects_per_image'] = (df_river_count['meter_count'] / df_river_count['image_count']).astype('int')

df_river_count['objects_per_image_str'] = df_river_count['objects_per_image'].astype('str')
df_river_count['meter_count_str'] = df_river_count['meter_count'].astype('str')
df_river_count['image_count_str'] = df_river_count['image_count'].astype('str')


# In[10]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

#initiate variables
color_dict = {'Cengkareng' : '#04C7E5', 'Ciliwung' : '#F43C7A', 'Grogol' : '#07344E', 
                        'Kaliadem' : '#578092', 'Marunda' : '#61E4B2', }

river_list = list(set(df_select['river']))

mapbox_access_token = 'pk.eyJ1IjoiY29saW52bCIsImEiOiJjamM3ZThhdjUxb2tjMnhtbTQ5MHF1OTh4In0.5Hw3JoFIdbjIcwovipXH6g'

aggregation_options = ['Minute aggregation', 'Hour aggregation']

#mapbox
mapbox_access_token = 'pk.eyJ1IjoiY29saW52bCIsImEiOiJjamM3ZThhdjUxb2tjMnhtbTQ5MHF1OTh4In0.5Hw3JoFIdbjIcwovipXH6g'

def get_scatter_mapbox(max_marker_size, hover_info, text, color):
     return go.Scattermapbox(
        lat=df_river_count['latitude'],
        lon=df_river_count['longitude'],
        customdata = df_river_count['river'],
        mode='markers',
        marker=go.scattermapbox.Marker(
            size=df_river_count['meter_count'],
            color =  color,
            sizeref = max(df_river_count['meter_count']) / (max_marker_size ** 2),
            opacity = 0.7,
            sizemode = 'area',
        ),
        text= text,
        hoverinfo = hover_info
     )

traces_mapbox = [
    get_scatter_mapbox(40, 'none', None, '#023857'),
    get_scatter_mapbox(32, 'text', df_river_count['river'] + '<br>Objects per image: ' + df_river_count['objects_per_image_str'], '#4DC4D8')
]

layout_mapbox = go.Layout(
#     title = 'Data collection points Jakarta',
    showlegend = False,
    autosize=True,
    hovermode='closest',
    clickmode = 'event+select',
    mapbox=dict(
        accesstoken=mapbox_access_token,
        bearing=0,
        center=dict(
            lat=np.mean(df_river_count['latitude']),
            lon=np.mean(df_river_count['longitude'])
        ),
        pitch=0,
        zoom=9,
        style = 'light'
    ),
    margin = go.layout.Margin(
        l = 20,
        r = 0
    )
)

#set layout of the app
app.layout = html.Div(children=[
    html.H2(children='Dashboard Jakarta'),

    html.Div(children='''
    '''),
    html.Div([
        dcc.Dropdown(
            id='aggregation-dropdown',
            options=[{'label': i, 'value': i} for i in aggregation_options],
            value='Minute aggregation'
        ),
    ],style={'width': '20%'}),
    html.Div(
        dcc.Graph(
            id='line-chart',
        ),style={'width': '68%', 'display': 'inline-block'}),
    html.Div(
        dcc.Graph(
            id='boxplot',
    ),style={'width': '28%', 'display': 'inline-block'}),
    
    html.Div(
        dcc.Graph(
            id='dotted-chart',
        ),style={'width': '68%', 'display': 'inline-block'}
    ),
    html.Div(
        dcc.Graph(
            id='mapbox',
            figure={
                'data': traces_mapbox,
                'layout': layout_mapbox
            }
    ),style={'width': '28%', 'display': 'inline-block'}),
#     html.Div([
#         dcc.Markdown(d("""
#             **Click Data**

#             Click on points in the graph.
#         """)),
#         html.Pre(id='click-data', style=styles['pre']),
#     ], className='three columns'),
 
])

@app.callback(
    Output('line-chart', 'figure'),
    [Input('aggregation-dropdown', 'value')])
def line_chart(aggregation_value):
    traces_linechart = []
    for i in range(1, len(session_list)):
        df_temp = df_select[df_select['session'] == session_list[i]]
        if aggregation_value == 'Hour aggregation':
            df_temp = df_temp.groupby('date-hour').sum().reset_index()
            date_variable = 'date-hour'
        else:
            date_variable = 'datetime'
            
        session_sum = df_temp['meter_count'].sum()
        river = list(df_select[df_select['session'] == i]['river'])
        river = river[0]

        if i in new_location_session:
            show_legend = True
        else:
            show_legend = False

        trace = go.Scatter(
            #they have to be aggregated to enable hour stuff, so we require different datasets
            x=df_temp[date_variable],
            y=df_temp['meter_count'],
            legendgroup = river,
            name = river,
            line = dict(color = color_dict[river]),
            mode = 'lines',
            showlegend = show_legend,
            opacity = 0.8,
            hoverinfo = 'text',
            text = 'Location: ' + river + '<br> Observation count: ' + df_select['meter_count_str'] + '<br> Session: ' + df_select['session_str'] + 
            '<br> Session count: ' + str(session_sum)
        )
        traces_linechart.append(trace)

        layout_linechart = dict(
            title = "Time series Jakarta all five locations",
            clickmode = 'event+select',
            margin = go.layout.Margin(
                t = 40
            ),
            xaxis = dict(
                rangeslider = dict(
                    visible = True
                ),
                zeroline = False,
                showgrid = False,
                range = [min(df_select[date_variable]),max(df_select[date_variable])]
            ),
            yaxis = dict(
                zeroline = False,
                range = [0,max(df_select[date_variable])],
                title = 'Number of plastic objects in image',
            ),
        )
    return {
        'data' : traces_linechart,
        'layout' : layout_linechart
    }


@app.callback(
    Output('dotted-chart', 'figure'),
    [Input('line-chart', 'relayoutData'),
     Input('mapbox', 'clickData')])
def dotted_chart(selection, click_data):
    try:
        x0 = selection["xaxis.range"][0]
        x1 = selection["xaxis.range"][1]
    except:
        x0 = min(df_select2['datetime'])
        x1 = max(df_select2['datetime'])
    
    river_value = 'Grogol'
    if click_data != None:
        river_value = click_data['points'][0]['customdata']
    df_grogol = df_select2[df_select2['river'] == river_value]

    
    return{
        'data' : [{"x": df_grogol['datetime'], 
                  "y": df_grogol['river+location'],
                  "marker": {"size": df_grogol['meter_count']*20, "opacity" : 0.3, "color": '#023857'}, 
                  "mode": "markers", 
                  "name": "Women", 
                  "text": df_grogol['meter_count_str'],
                  "type": "scatter",
                  "hovermode": "closest"
        }],

        'layout' : {
            "margin": go.layout.Margin(l = 150),
            "title": "Plastic count observations distributed over locations at " + river_value + 
            ". <br> <i> Double click on any of the markers in the map to change the location. </i>", 
            "xaxis": dict(title = "Time range", range = [x0, x1])
            }
        }

@app.callback(
    Output('boxplot', 'figure'),
    [Input('line-chart', 'relayoutData'),
    ])

def boxplot(selection):
    try:
        x0 = selection["xaxis.range"][0]
        x1 = selection["xaxis.range"][1]
    except:
        x0 = min(df_select['datetime'])
        x1 = max(df_select['datetime'])
    
    df_select3 = df_select[df_select['datetime'] > x0]
    df_select3 = df_select3[df_select3['datetime'] < x1]
    
    traces_boxplot = []
    for i in range(len(river_list)):
        df_temp = df_select3[df_select3['river'] == river_list[i]]
        trace = go.Box(
            y=df_temp['meter_count'],
            name = river_list[i],
            marker = dict(
                color = color_dict[river_list[i]],
            )
        )
        traces_boxplot.append(trace)

    layout_boxplot = go.Layout(
        showlegend = False,
        yaxis = dict(range = [0, max(df_select['meter_count'])]),
        margin = go.layout.Margin(
            l = 40,
            t= 50,
            r = 0
        )
    )

    return{
        'data' : traces_boxplot,
        'layout' : layout_boxplot
    }
# @app.callback(
#     Output('click-data', 'children'),
#     [Input('mapbox', 'clickData')])
# def display_click_data(clickData):
#     return json.dumps(clickData, indent=2)

if __name__ == '__main__':
    app.run_server(debug=False)

