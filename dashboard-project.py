import dash
from dash import html
from dash import dcc
from dash.dependencies import Input, Output
import pandas as pd
import plotly.express as px
import pickle
from sklearn import  metrics
import numpy as np

#Define CSS style
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

#Display raw data weather data
df = pd.read_csv('SN69100_2015-2019_daily.csv')
df['date'] = pd.to_datetime(df['date']) # create a new column 'data time' of datetime type
fig1 = px.line(df, x="date", y=df.columns[:])# Creates a figure with the raw data

#Display raw Power data
df = pd.read_csv('Norway_Power_2015-2019_daily.csv')
df['date'] = pd.to_datetime(df['date']) # create a new column 'data time' of datetime type
fig2 = px.line(df, x="date", y=df.columns[:])# Creates a figure with the raw data

df = pd.read_csv('SN69100_2020_daily.csv')
df['date'] = pd.to_datetime(df['date']) # create a new column 'data time' of datetime type
df = df.set_index('date', drop=True) # make 'datetime' into index
df['month'] = df.index.month
df = df.loc['2020-01-01':'2020-07-31']
X2=df.values

df_real = pd.read_csv('Norway_Power_2020_daily.csv')
y2=df_real['power_MW'].values

#Load RF model
with open('RF_model_final_prosj.pkl','rb') as file:
    RF_model2=pickle.load(file)

y2_pred_RF = RF_model2.predict(X2)

#Evaluate errors
MAE_RF=metrics.mean_absolute_error(y2,y2_pred_RF)
MBE_RF=np.mean(y2-y2_pred_RF) 
MSE_RF=metrics.mean_squared_error(y2,y2_pred_RF)  
RMSE_RF= np.sqrt(metrics.mean_squared_error(y2,y2_pred_RF))
cvRMSE_RF=RMSE_RF/np.mean(y2)
NMBE_RF=MBE_RF/np.mean(y2)

# Create data frames with predictin results and error metrics 
d = {'Methods': ['Random Forest'], 'MAE': [MAE_RF],'MBE': [MBE_RF], 'MSE': [MSE_RF], 'RMSE': [RMSE_RF],'cvMSE': [cvRMSE_RF],'NMBE': [NMBE_RF]}
df_metrics = pd.DataFrame(data=d)
d={'date':df_real['date'].values, 'RandomForest': y2_pred_RF}
df_forecast=pd.DataFrame(data=d)

# merge real and forecast results and creates a figure with it
df_results=pd.merge(df_real,df_forecast, on='date')

fig3 = px.line(df_results,x=df_results.columns[0],y=df_results.columns[1:4])

# Define auxiliary functions
def generate_table(dataframe, max_rows=10):
    return html.Table([
        html.Thead(
            html.Tr([html.Th(col) for col in dataframe.columns])
        ),
        html.Tbody([
            html.Tr([
                html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
            ]) for i in range(min(len(dataframe), max_rows))
        ])
    ])


app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

app.layout = html.Div([
    html.H1('Norway Energy Consumtion (MWh)'),
    html.P('Using data from 2015-2019 to forecast energy consumtion in Norway for 2020.'),
    dcc.Tabs(id='tabs', value='tab-1', children=[
        dcc.Tab(label='Raw Data', value='tab-1'),
        dcc.Tab(label='Forecast', value='tab-2'),
        dcc.Tab(label='Error Metrics', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

@app.callback(Output('tabs-content', 'children'),
              Input('tabs', 'value'))

def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('Raw weather and power data 2015-2019'),
            dcc.Graph(
                id='wather-data',
                figure=fig1,
            ),
            dcc.Graph(
                id='power-data',
                figure=fig2,
            ),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H4('Predict Power 01.01.2020 to 31.07.2020 (MWh)'),
            dcc.Graph(
                id='predict-data',
                figure=fig3,
                ),
            
        ])
    elif tab == 'tab-3':
        return html.Div([
            html.H4('Error Metrics for Forecast 2020'),
                        generate_table(df_metrics)
        ])


if __name__ == '__main__':
    app.run_server()
