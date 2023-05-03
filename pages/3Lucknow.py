import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler

import keras
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense, LSTM

import streamlit as st

st.title("Lucknow")

data = pd.read_csv("./Datasets/Lucknow.csv")
# st.markdown("#### Data:")

col1, col2 = st.columns(2, gap="small")
col1.markdown("#### Head of Data:")
col1.dataframe(data.head())
col2.markdown("#### Tail of Data:")
col2.dataframe(data.tail())

data.fillna(method='ffill', inplace=True)
data.dropna(axis=0, inplace=True)

st.markdown("#### YTD Plot:")
st.line_chart(data['tavg'])

ma_day = [30, 180, 360]
for ma in ma_day:
    column_name = f"MA for {ma} days"
    data[column_name] = data['tavg'].rolling(ma).mean()

st.markdown("#### Moving Average Plot:")
col1, col2, col3 = st.columns(3, gap="large")

col1.markdown(f'{ma_day[0]} DAYS MA')
col1.line_chart(data[[f'MA for {ma_day[0]} days']])

col2.markdown(f'{ma_day[1]} DAYS MA')
col2.line_chart(data[[f'MA for {ma_day[1]} days']])

col3.markdown(f'{ma_day[2]} DAYS MA')
col3.line_chart(data[[f'MA for {ma_day[2]} days']])

# Create a new dataframe with only the 'Close column 

with st.spinner('PREPROCESSING THE DATA'):
    last = 30
        
    training_data_len = data[data['time'] == '01-01-2022']
    training_data_len = 11688

    data = data.filter(['tavg'])
    # Convert the dataframe to a numpy array
    dataset = data.values

    # Scale the data
    scaler = MinMaxScaler(feature_range=(0,1))
    scaled_data = scaler.fit_transform(dataset)

    # Create the training data set 
    # Create the scaled training data set
    train_data = scaled_data[0:int(training_data_len), :]
    # Split the data into x_train and y_train data sets
    x_train = []
    y_train = []

    for i in range(last, len(train_data)):
        x_train.append(train_data[i-last:i, 0])
        y_train.append(train_data[i, 0])

    # Convert the x_train and y_train to numpy arrays 
    x_train, y_train = np.array(x_train), np.array(y_train)

    # Reshape the data
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

st.checkbox("PREPROCESSED THE DATA", value=True, disabled=True)

# Build the LSTM model
with st.spinner('TRAINING THE DEEP LEARNING MODEL (Could take about couple of minutes)'):
    model = Sequential()
    model.add(LSTM(128, return_sequences=True, input_shape= (x_train.shape[1],1)))
    model.add(LSTM(64, return_sequences=False))
    model.add(Dense(25))
    model.add(Dense(1))

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    model.fit(x_train, y_train, epochs=3, verbose=1)

st.checkbox("TRAINED THE MODEL", value=True, disabled=True)

with st.spinner('TESTING THE MODEL'):
    # Create the testing data set
    # Create a new array containing scaled values
    test_data = scaled_data[(int(training_data_len) - last): , :]
    # print(test_data.shape)

    # Create the data sets x_test and y_test
    x_test = []
    y_test = dataset[training_data_len:, :]
    for i in range(last, len(test_data)):
        x_test.append(test_data[i-last:i, 0])

    # Convert the data to a numpy array
    x_test = np.array(x_test)
    # print(x_test.shape)

    # Reshape the data
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

    # Get the models predicted price values 
    predictions = model.predict(x_test)
    predictions = scaler.inverse_transform(predictions)

# Get the root mean squared error (RMSE)
rmse = np.sqrt(np.mean(((predictions - y_test) ** 2)))
st.checkbox("TESTING THE MODEL", value=True, disabled=True)

st.subheader('PREDICTING THE TEMPERATURE')
preds = [i[0] for i in predictions]

# Plot the data
train = data.iloc[11500:training_data_len, :]
valid = data.iloc[training_data_len:, :]
# print(valid)

valid['Predictions'] = preds
valid = valid.sort_index(ascending=True)
# print(valid)

# Visualize the data
plt.figure(figsize=(20, 6))

plt.title('Model')
plt.xlabel('Date', fontsize=18)
plt.ylabel('Avg. Temp', fontsize=18)

data = pd.DataFrame({'Train':train['tavg'], 'Val':valid['tavg'], 'Predictions':valid['Predictions']})
st.line_chart(data)

st.markdown(f"##### RMSE Score = {round(rmse, 4)}")
# st.line_chart(train['tavg'])
# st.line_chart(valid['tavg'])
# st.line_chart(valid['Predictions'])

# plt.plot(valid[['tavg', 'Predictions']])

# plt.legend(['Train', 'Val', 'Predictions'], loc='best')
# plt.show()