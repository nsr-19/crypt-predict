import streamlit as st
import pandas as pd
import numpy as np
import pandas_datareader as data

# Importinng Libraries for Data Analysis
from sklearn.metrics import mean_squared_error, mean_absolute_error, explained_variance_score, r2_score 
from sklearn.metrics import mean_poisson_deviance, mean_gamma_deviance, accuracy_score
from sklearn.preprocessing import MinMaxScaler

# Importinng Libraries for building Model
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.layers import LSTM
from keras.models import load_model

# Importing Libaraies for Plotting Data
import matplotlib.pyplot as plt
from itertools import cycle
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots


# Importing 'yfinance' libaray to use Yahoo Finance API
import yfinance as yf

# Pre-processing Function
def remove(x):
    """
    This function will strip the data column of the dataframe.
    """
    x = str(x)
    res = x.split(" ")[0]
    return res

# Title
st.title('Cryptocurrency Price Predictor')

#User Input
user_input = st.text_input('Enter Stock Ticker and Press Enter', 'BTC-USD')
#user_input = st.selectbox(
#    'Choose your Cryptocurrency',
#    ('BTC-USD', 'ETH-USD', 'RPOWER.NS'))

#st.write('You selected:', user_input)

# Calling Yahoo Finance API for fetching Dataset
btc = yf.Ticker(user_input)
btc = btc.history(period = "max")  # we need max data available 
btc.index = pd.to_datetime(btc.index)  # changing the index
btc.index = btc.index.to_series().apply(lambda x:remove(x))  # applying preprocessing function
btc.to_csv('data.csv')  # saving the data in csv format

st.subheader(user_input + ' Dataset')
btc  # displaying the data.

df = pd.read_csv('data.csv') # storing data in CSV file

st.subheader(user_input + ' Dataset Description')
st.write(df.describe())

# Start date and End date of the dataset
sd = df.iloc[0][0]
ed = df.iloc[-1][0]

# Analysis from 2014 to Today
df['Date'] = pd.to_datetime(df['Date'], format='%Y-%m-%d')
y_overall = df.loc[(df['Date'] >= sd) & (df['Date'] <= ed)]
y_overall.drop(y_overall[['Dividends','Stock Splits', 'Volume']],axis=1)
monthwise = y_overall.groupby(y_overall['Date'].dt.strftime('%B'))[['Open','Close']].mean()
new_order = ['January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 
             'September', 'October', 'November', 'December']
monthwise = monthwise.reindex(new_order, axis=0)
#monthwise

names = cycle(['BTC Open Price','BTC Close Price','BTC High Price','BTC Low Price'])
fig = px.line(y_overall, x=y_overall.Date, y=[y_overall['Open'], y_overall['Close'], 
                                          y_overall['High'], y_overall['Low']],
             labels={'Date': 'Year','value':'BTC Value (in USD)'})
fig.update_layout(title_text= user_input +' Analysis Chart', font_size=15, font_color='black',legend_title_text='Stock Parameters')
fig.for_each_trace(lambda t:  t.update(name = next(names)))
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)

st.subheader(user_input + ' Price Chart')
st.plotly_chart(fig) #fig.show()

# Let's first take all the Close Prices 
closedf = df[['Date','Close']]
st.write("Shape of close dataframe:", closedf.shape)

fig = px.line(closedf, x = closedf.Date, y = closedf.Close, labels = {'date':'Year','close':'Close Price (USD)'})
fig.update_traces(marker_line_width = 2, opacity = 0.8, marker_line_color = 'orange')
fig.update_layout(title_text = 'BTC Close Price: 2014-2023', plot_bgcolor='white', 
                  font_size = 15, font_color = 'black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
st.plotly_chart(fig)

st.subheader('Model Building')

# Taking data for building model
mb_date = st.text_input('Enter the starting date to be considered for building prediction model', '2022-06-20')


closedf = closedf[closedf['Date'] > mb_date]
close_stock = closedf.copy()
st.write("Total data for prediction: ", closedf.shape[0])

fig = px.line(closedf, x = closedf.Date, y = closedf.Close, labels = {'date':'Date','close':'Close Price'})
fig.update_traces(marker_line_width = 2, opacity = 0.8, marker_line_color = 'orange')
fig.update_layout(title_text = 'Considered period to predict '+ user_input+ 'close price', plot_bgcolor = 'white', font_size = 15, font_color = 'black')
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
st.plotly_chart(fig)

# Normalizing Data
del closedf['Date'] # deleting date column and normalizing using MinMax Scaler
scaler = MinMaxScaler(feature_range=(0,1))
closedf = scaler.fit_transform(np.array(closedf).reshape(-1,1))
#print(closedf.shape)

# Splitting Data into Training Set and Testing Set
training_size = int(len(closedf)*0.80) # Taking 80% data for Training Model
test_size = len(closedf)-training_size
train_data,test_data = closedf[0:training_size,:],closedf[training_size:len(closedf),:1]
st.write("Train Data Shape: ", train_data.shape)
st.write("Test Data Shape: ", test_data.shape)

# Time Series Analysis

# Convert an array of values into a matrix of dataset
def create_dataset(dataset, time_step = 1):
    dataX, dataY = [], []
    for i in range(len(dataset)-time_step-1):
        a = dataset[i:(i+time_step), 0]   ###i=0, 0,1,2,3-----99   100 
        dataX.append(a)
        dataY.append(dataset[i + time_step, 0])
    return np.array(dataX), np.array(dataY)

time_step = 15
X_train, Y_train = create_dataset(train_data, time_step)
X_test, Y_test = create_dataset(test_data, time_step)

#print("X_train: ", X_train.shape)
#print("Y_train: ", Y_train.shape)
#print("X_test: ", X_test.shape)
#print("Y_test", Y_test.shape)

# Reshape input to be [samples, time steps, features] which is required for LSTM

X_train = X_train.reshape(X_train.shape[0],X_train.shape[1] , 1)
X_test = X_test.reshape(X_test.shape[0],X_test.shape[1] , 1)
#print("X_train: ", X_train.shape)
#print("X_test: ", X_test.shape)

# Model Building
model = Sequential()
model.add(LSTM(10, input_shape = (None,1), activation = "relu"))
model.add(Dense(1))
model.compile(loss = "mean_squared_error", optimizer = "adam")

#model = load_model('keras_model.h5')

history = model.fit(X_train, Y_train, validation_data = (X_test, Y_test), epochs = 350, batch_size = 25, verbose = 1)

# Plotting Loss vs Validation Loss

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(loss))

fig2 = plt.figure()
plt.plot(epochs, loss, 'r', label = 'Training loss')
plt.plot(epochs, val_loss, 'b', label = 'Validation loss')
plt.title('Training and validation loss')
plt.legend(loc = 0)
plt.figure()

st.subheader('Training and Validation Loss for above built model')
st.pyplot(fig2)

st.subheader('Model Performance Metrics Checks')

# Prediction and Performance Metrics Checks
train_predict = model.predict(X_train)
test_predict = model.predict(X_test)
train_predict.shape, test_predict.shape

# Transform back to original form
train_predict = scaler.inverse_transform(train_predict)
test_predict = scaler.inverse_transform(test_predict)
original_Ytrain = scaler.inverse_transform(Y_train.reshape(-1,1)) 
original_Ytest = scaler.inverse_transform(Y_test.reshape(-1,1)) 

# Evaluation Metrics: RMSE, MSE and MAE

import math

st.caption('MSE, RMSE and MAE')
st.write("Train data MSE: ", mean_squared_error(original_Ytrain, train_predict))
st.write("Train data RMSE: ", math.sqrt(mean_squared_error(original_Ytrain, train_predict)))
st.write("Train data MAE: ", mean_absolute_error(original_Ytrain, train_predict))
st.write("-------------------------------------------------------------------------------------")
st.write("Test data RMSE: ", math.sqrt(mean_squared_error(original_Ytest, test_predict)))
st.write("Test data MSE: ", mean_squared_error(original_Ytest, test_predict))
st.write("Test data MAE: ", mean_absolute_error(original_Ytest, test_predict))

# Varianace Regression Score
st.caption('Varianace Regression Score')
st.write("Train data explained variance regression score:", explained_variance_score(original_Ytrain, train_predict))
st.write("Test data explained variance regression score:", explained_variance_score(original_Ytest, test_predict))

# R Square score for regression
st.caption('R Square score for regression')
st.write("Train data R2 score:", r2_score(original_Ytrain, train_predict))
st.write("Test data R2 score:", r2_score(original_Ytest, test_predict))

# Regression Loss: Mean Gamma deviance regression loss (MGD) and Mean Poisson deviance regression loss (MPD)
st.caption('Regression Loss: Mean Gamma deviance regression loss (MGD) and Mean Poisson deviance regression loss (MPD)')
st.write("Train data MGD: ", mean_gamma_deviance(original_Ytrain, train_predict))
st.write("Test data MGD: ", mean_gamma_deviance(original_Ytest, test_predict))
st.write("----------------------------------------------------------------------")
st.write("Train data MPD: ", mean_poisson_deviance(original_Ytrain, train_predict))
st.write("Test data MPD: ", mean_poisson_deviance(original_Ytest, test_predict))

# Predicting Next 15 days
x_input = test_data[len(test_data)-time_step:].reshape(1,-1)
temp_input = list(x_input)
temp_input = temp_input[0].tolist()

from numpy import array

lst_output = []
n_steps = time_step
i = 0
pred_days = 30
while(i<pred_days):
    
    if(len(temp_input)>time_step):
        
        x_input = np.array(temp_input[1:])
        #print("{} day input {}".format(i,x_input))
        x_input = x_input.reshape(1,-1)
        x_input = x_input.reshape((1, n_steps, 1))
        
        yhat = model.predict(x_input, verbose=0)
        #print("{} day output {}".format(i,yhat))
        temp_input.extend(yhat[0].tolist())
        temp_input = temp_input[1:]
        #print(temp_input)
       
        lst_output.extend(yhat.tolist())
        i = i+1
        
    else:
        
        x_input = x_input.reshape((1, n_steps,1))
        yhat = model.predict(x_input, verbose=0)
        temp_input.extend(yhat[0].tolist())
        
        lst_output.extend(yhat.tolist())
        i = i+1
               
#st.write("Output of predicted next days: ", len(lst_output))

# shift train predictions for plotting

look_back = time_step
trainPredictPlot = np.empty_like(closedf)
trainPredictPlot[:, :] = np.nan
trainPredictPlot[look_back:len(train_predict) + look_back, :] = train_predict
#print("Train predicted data: ", trainPredictPlot.shape)

# shift test predictions for plotting
st.subheader('Comparision between original close price vs predicted close price')
testPredictPlot = np.empty_like(closedf)
testPredictPlot[:, :] = np.nan
testPredictPlot[len(train_predict)+(look_back*2)+1:len(closedf)-1, :] = test_predict
#print("Test predicted data: ", testPredictPlot.shape)

names = cycle(['Original close price','Train predicted close price','Test predicted close price'])


plotdf = pd.DataFrame({'date': close_stock['Date'],
                       'original_close': close_stock['Close'],
                      'train_predicted_close': trainPredictPlot.reshape(1,-1)[0].tolist(),
                      'test_predicted_close': testPredictPlot.reshape(1,-1)[0].tolist()})

fig = px.line(plotdf,x=plotdf['date'], y = [plotdf['original_close'],plotdf['train_predicted_close'],
                                          plotdf['test_predicted_close']],
              labels={'value':user_input +' (in USD)','date': 'Date'})
fig.update_layout(title_text = 'Comparision between original close price vs predicted close price',
                  plot_bgcolor = 'white', font_size = 15, font_color = 'black', legend_title_text = 'Close Price')
fig.for_each_trace(lambda t:  t.update(name = next(names)))

fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
st.plotly_chart(fig)
#fig.show()

# Plotting for next 15 days

last_days = np.arange(1,time_step + 1)
day_pred = np.arange(time_step + 1, time_step + pred_days + 1)
#st.write(last_days)
#st.write(day_pred)

temp_mat = np.empty((len(last_days) + pred_days + 1,1))
temp_mat[:] = np.nan
temp_mat = temp_mat.reshape(1,-1).tolist()[0]

last_original_days_value = temp_mat
next_predicted_days_value = temp_mat

last_original_days_value[0:time_step + 1] = scaler.inverse_transform(closedf[len(closedf)-time_step:]).reshape(1,-1).tolist()[0]
next_predicted_days_value[time_step + 1:] = scaler.inverse_transform(np.array(lst_output).reshape(-1,1)).reshape(1,-1).tolist()[0]

new_pred_plot = pd.DataFrame({
    'last_original_days_value':last_original_days_value,
    'next_predicted_days_value':next_predicted_days_value
})

names = cycle(['Last 15 days close price','Predicted next 30 days close price'])

fig = px.line(new_pred_plot, x = new_pred_plot.index, y = [new_pred_plot['last_original_days_value'],
                                                      new_pred_plot['next_predicted_days_value']],
              labels = {'value':user_input + ' (in USD)','index': 'Timestamp'})
fig.update_layout(title_text = 'Compare last 15 days vs next 30 days',
                  plot_bgcolor = 'white', font_size = 15, font_color = 'black', legend_title_text = 'Close Price')

fig.for_each_trace(lambda t: t.update(name = next(names)))
fig.update_xaxes(showgrid = False)
fig.update_yaxes(showgrid = False)
st.plotly_chart(fig)