#!/usr/bin/env python
# coding: utf-8

# In[56]:


import urllib.request

# URLs to download
url_csv = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/datasets/spacex_launch_dash.csv"
url_app = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-DS0321EN-SkillsNetwork/labs/module_3/spacex_dash_app.py"

# File names to save
filename_csv = "spacex_launch_dash.csv"
filename_app = "spacex_dash_app.py"

# Download CSV
urllib.request.urlretrieve(url_csv, filename_csv)

# Download Python file
urllib.request.urlretrieve(url_app, filename_app)

print("Files downloaded successfully!")


# In[64]:


import pandas as pd

# File names
filename_csv = "spacex_launch_dash.csv"
filename_app = "spacex_dash_app.py"

# Read CSV file into a DataFrame
spacex_df = pd.read_csv(filename_csv)

# Display the DataFrame
print("Contents of", filename_csv)
print(spacex_df.head())  # Display the first few rows

# Read the contents of the Python file
with open(filename_app, "r") as file:
    app_contents = file.read()

# Display the contents of the Python file
print("\nContents of", filename_app)
print(app_contents)


# In[66]:


import dash_core_components as dcc

# Get unique launch site names from the DataFrame
launch_sites = spacex_df['Launch Site'].unique()

# Create options for the dropdown
dropdown_options = [{'label': site, 'value': site} for site in launch_sites]

# Add an option for 'All Sites'
dropdown_options.insert(0, {'label': 'All Sites', 'value': 'ALL'})

# Define the Dropdown component
dcc.Dropdown(
    id='site-dropdown',
    options=dropdown_options,
    value='ALL',  # Default selected value
    placeholder="Select a Launch Site",
    searchable=True  # Allow searching for options
)


# In[72]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Assuming spacex_df is your DataFrame containing the SpaceX launch data
# Replace spacex_df with your actual DataFrame if different
# Assuming spacex_df is your DataFrame containing the SpaceX launch data
# Replace spacex_df with your actual DataFrame if different
spacex_df = pd.read_csv("spacex_launch_dash.csv")


# Get unique launch site names from the DataFrame
launch_sites = spacex_df['Launch Site'].unique()

# Create options for the dropdown
dropdown_options = [{'label': site, 'value': site} for site in launch_sites]

# Add an option for 'All Sites'
dropdown_options.insert(0, {'label': 'All Sites', 'value': 'ALL'})

app = dash.Dash(__name__)

app.layout = html.Div([
    # Dropdown for selecting launch site
    dcc.Dropdown(
        id='site-dropdown',
        options=dropdown_options,
        value='ALL',  # Default selected value
        placeholder="Select a Launch Site",
        searchable=True  # Allow searching for options
    ),
    
    # Pie chart to display success ratio
    dcc.Graph(id='success-pie-chart')
])

@app.callback(
    Output('success-pie-chart', 'figure'),
    [Input('site-dropdown', 'value')]
)
def update_pie_chart(selected_site):
    if selected_site == 'ALL':
        # Create a pie chart for all sites (total success rate)
        fig = px.pie(spacex_df, names='class', title='Total Launch Success Rate')
    else:
        # Filter the DataFrame to include data for the selected site
        filtered_df = spacex_df[spacex_df['Launch Site'] == selected_site]
        
        # Count success (class=1) and failure (class=0) launches for the selected site
        success_count = len(filtered_df[filtered_df['class'] == 1])
        failure_count = len(filtered_df[filtered_df['class'] == 0])
        
        # Create a pie chart to show success and failure counts for the selected site
        labels = ['Success', 'Failure']
        values = [success_count, failure_count]
        fig = px.pie(values=values, names=labels, title=f'Success-Failure Ratio for {selected_site}')
    
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8060)  # Use a different port, e.g., 8060


# In[74]:


import dash_core_components as dcc

# Define the RangeSlider component
dcc.RangeSlider(
    id='payload-slider',
    min=0,  # Slider starting point (in Kg)
    max=10000,  # Slider ending point (in Kg)
    step=1000,  # Slider interval (in Kg)
    value=[0, 10000]  # Current selected range (in Kg)
)


# In[75]:


@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [Input('site-dropdown', 'value'), Input('payload-slider', 'value')]
)
def update_scatter_chart(selected_site, payload_range):
    if selected_site == 'ALL':
        # Filter the DataFrame to include data within the selected payload range
        filtered_df = spacex_df[(spacex_df['Payload Mass (kg)'] >= payload_range[0]) & 
                                (spacex_df['Payload Mass (kg)'] <= payload_range[1])]
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class', color='Booster Version Category',
                         title='Payload vs. Launch Outcome (All Sites)')
    else:
        # Filter the DataFrame to include data for the selected site and within the selected payload range
        filtered_df = spacex_df[(spacex_df['Launch Site'] == selected_site) &
                                (spacex_df['Payload Mass (kg)'] >= payload_range[0]) & 
                                (spacex_df['Payload Mass (kg)'] <= payload_range[1])]
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class', color='Booster Version Category',
                         title=f'Payload vs. Launch Outcome for {selected_site}')
    
    return fig


# In[77]:


import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import plotly.express as px

# Load the DataFrame 'spacex_df'
# Replace with your actual DataFrame or load the data here
# spacex_df = ...

# Get unique launch site names from the DataFrame
launch_sites = spacex_df['Launch Site'].unique()

# Create options for the dropdown
dropdown_options = [{'label': site, 'value': site} for site in launch_sites]
dropdown_options.insert(0, {'label': 'All Sites', 'value': 'ALL'})

app = dash.Dash(__name__)

app.layout = html.Div([
    dcc.Dropdown(
        id='site-dropdown',
        options=dropdown_options,
        value='ALL',
        placeholder="Select a Launch Site",
        searchable=True
    ),
    
    dcc.RangeSlider(
        id='payload-slider',
        min=0,
        max=10000,
        step=1000,
        value=[0, 10000]
    ),
    
    dcc.Graph(id='success-pie-chart'),
    dcc.Graph(id='success-payload-scatter-chart')
])

@app.callback(
    Output('success-pie-chart', 'figure'),
    [Input('site-dropdown', 'value')]
)
def update_pie_chart(selected_site):
    if selected_site == 'ALL':
        fig = px.pie(spacex_df, names='class', title='Total Launch Success Rate')
    else:
        filtered_df = spacex_df[spacex_df['Launch Site'] == selected_site]
        success_count = len(filtered_df[filtered_df['class'] == 1])
        failure_count = len(filtered_df[filtered_df['class'] == 0])
        labels = ['Success', 'Failure']
        values = [success_count, failure_count]
        fig = px.pie(values=values, names=labels, title=f'Success-Failure Ratio for {selected_site}')
    return fig

@app.callback(
    Output('success-payload-scatter-chart', 'figure'),
    [Input('site-dropdown', 'value'), Input('payload-slider', 'value')]
)
def update_scatter_chart(selected_site, payload_range):
    if selected_site == 'ALL':
        filtered_df = spacex_df[(spacex_df['Payload Mass (kg)'] >= payload_range[0]) & 
                                (spacex_df['Payload Mass (kg)'] <= payload_range[1])]
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class', color='Booster Version Category',
                         title='Payload vs. Launch Outcome (All Sites)')
    else:
        filtered_df = spacex_df[(spacex_df['Launch Site'] == selected_site) &
                                (spacex_df['Payload Mass (kg)'] >= payload_range[0]) & 
                                (spacex_df['Payload Mass (kg)'] <= payload_range[1])]
        fig = px.scatter(filtered_df, x='Payload Mass (kg)', y='class', color='Booster Version Category',
                         title=f'Payload vs. Launch Outcome for {selected_site}')
    return fig

if __name__ == '__main__':
    app.run_server(debug=True, port=8060)  # Use a different port, e.g., 8060



# In[ ]:




