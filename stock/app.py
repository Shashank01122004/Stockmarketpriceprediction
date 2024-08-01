import numpy as np
import pandas as pd
import yfinance as yf
from keras.models import load_model     # To load the pre-trained LSTM model
import streamlit as st
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

# Load the pre-trained model
model = load_model('stock_price_prediction_model.keras')

# Setup Streamlit interface
st.header('Stock Market Predictor')

# Get stock symbol input from user
stock = st.text_input('Enter Stock Symbol', 'GOOG')
start = '2012-01-01'
end = '2022-12-31'

# Download stock data using yfinance
data = yf.download(stock, start, end)

# Display stock data
st.subheader('Stock Data')
st.write(data)

# Prepare training and testing data
data_train = pd.DataFrame(data.Close[0: int(len(data) * 0.80)])
data_test = pd.DataFrame(data.Close[int(len(data) * 0.80): len(data)])

scaler = MinMaxScaler(feature_range=(0, 1))
pas_150_days = data_train.tail(150)
data_test = pd.concat([pas_150_days, data_test], ignore_index=True)
data_test_scale = scaler.fit_transform(data_test)

# Plot Price vs MA75
st.subheader('Price vs MA75')
ma_75_days = data.Close.rolling(75).mean()
fig1 = plt.figure(figsize=(8, 6))
plt.plot(ma_75_days, 'r')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig1)

# Plot Price vs MA75 vs MA150
st.subheader('Price vs MA75 vs MA150')
ma_150_days = data.Close.rolling(150).mean()
fig2 = plt.figure(figsize=(8, 6))
plt.plot(ma_75_days, 'r')
plt.plot(ma_150_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig2)

# Plot Price vs MA150 vs MA200
st.subheader('Price vs MA150 vs MA200')
ma_200_days = data.Close.rolling(200).mean()
fig3 = plt.figure(figsize=(8, 6))
plt.plot(ma_150_days, 'r')
plt.plot(ma_200_days, 'b')
plt.plot(data.Close, 'g')
plt.show()
st.pyplot(fig3)

# Prepare test data for prediction
x = []
y = []
for i in range(100, data_test_scale.shape[0]):
    x.append(data_test_scale[i - 100:i])
    y.append(data_test_scale[i, 0])

x, y = np.array(x), np.array(y)

# Predict stock prices
predict = model.predict(x)

# Inverse scaling to get back original values
scale = 1 / scaler.scale_[0]
predict = predict * scale
y = y * scale

# Plot Original Price vs Predicted Price
st.subheader('Original Price vs Predicted Price')
fig4 = plt.figure(figsize=(8, 6))
plt.plot(predict, 'r', label='Predicted Price')
plt.plot(y, 'g', label='Actual Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()
st.pyplot(fig4)

# Calculate error
error = y - predict.reshape(-1)

# Plot error graph
st.subheader('Error in Predicted vs Actual Prices')
fig5 = plt.figure(figsize=(8, 6))
plt.plot(error, label='Error', color='blue')
plt.xlabel('Time')
plt.ylabel('Error')
plt.legend()
plt.show()
st.pyplot(fig5)