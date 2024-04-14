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
external_stylesheets = ['https://stackpath.bootstrapcdn.com/bootswatch/4.5.2/slate/bootstrap.min.css']

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

#Load and run LR model
with open('LR_model_final_prosj.pkl','rb') as file:
    LR_model2=pickle.load(file)
y2_pred_LR = LR_model2.predict(X2)

#Evaluate errors
MAE_LR=metrics.mean_absolute_error(y2,y2_pred_LR) 
MBE_LR=np.mean(y2-y2_pred_LR)
MSE_LR=metrics.mean_squared_error(y2,y2_pred_LR)  
RMSE_LR= np.sqrt(metrics.mean_squared_error(y2,y2_pred_LR))
cvRMSE_LR=RMSE_LR/np.mean(y2)
NMBE_LR=MBE_LR/np.mean(y2)

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
d = {'Methods': ['Linear Regression','Random Forest'], 'MAE': [MAE_LR, MAE_RF],'MBE': [MBE_LR, MBE_RF], 'MSE': [MSE_LR, MSE_RF], 'RMSE': [RMSE_LR, RMSE_RF],'cvMSE': [cvRMSE_LR, cvRMSE_RF],'NMBE': [NMBE_LR, NMBE_RF]}
df_metrics = pd.DataFrame(data=d)
d={'date':df_real['date'].values, 'LinearRegression': y2_pred_LR,'RandomForest': y2_pred_RF}
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

# Define app layout
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)

app.layout = html.Div(style={
    'background-image': 'url("/assets/background.jpg")',
    'background-size': 'cover',
    'background-repeat': 'no-repeat',
    'background-position': 'center center',
    'background-attachment': 'fixed',
    'min-height': '100vh',  # This ensures that the background covers the entire height of the view
    'width': '100%'
}, children=[
    html.Div([
        html.H1('Power Consumption in Norway', className='display-4 text-center mt-4 mb-4', style={'color': 'white'}),
        html.P('Welcome! This dashboard provides a forecast of daily power consumption in Norway based on weather data. Navigate through the tabs to explore the raw data and forecasted results. The "Raw data" tab displays historical temperature, humidity, and Power usage from 2015 to 2020, while the "Forecast" tab shows the forecasted energy consumption using machine learning models.', 
               className='lead text-center', style={'color': 'white', 'font-size': '20px'}),
    ], className='container', style={'text-shadow': '2px 2px 4px #000000'}),  # text-shadow to improve text visibility
    html.Div([
        dcc.Tabs(id='tabs', value='tab-1', children=[
            dcc.Tab(label='Raw data', value='tab-1', className='custom-tab'),
            dcc.Tab(label='Power consumption forecast', value='tab-2', className='custom-tab'),
        ], className='nav nav-pills nav-fill flex-column flex-md-row'),
    ], className='container'),
    html.Div(id='tabs-content', className='container')
])
# Define callback to update content based on selected tab
@app.callback(Output('tabs-content', 'children'), [Input('tabs', 'value')])
def render_content(tab):
    if tab == 'tab-1':
        return html.Div([
            html.H4('', className='text-center mb-4 text-light'),
            dcc.Graph(
                id='weather-data',
                figure=fig1,
            ),
            dcc.Graph(
                id='power-data',
                figure=fig2,
            ),
        ])
    elif tab == 'tab-2':
        return html.Div([
            html.H4('', className='text-center mb-4 text-light'),
            dcc.Graph(
                id='forecast-data',
                figure = fig3,
            ),
            html.Hr(),
            html.H4('Error Metrics', className='text-center mb-4 text-light'),
            html.Div([
                html.Table(
                    [html.Tr([html.Th(col) for col in df_metrics.columns])] +
                    [html.Tr([html.Td(df_metrics.iloc[i][col]) for col in df_metrics.columns]) for i in range(len(df_metrics))]
                )
            ], className='table table-striped table-hover text-dark', style={'background-color': 'white'})
        ])

if __name__ == '__main__':
    app.run_server(debug=True)