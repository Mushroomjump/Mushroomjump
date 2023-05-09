import datetime
import math
import matplotlib.pyplot as pl
import numpy as np
import requests
import streamlit as st
import yfinance as yf
from newsapi import NewsApiClient
from nltk.sentiment import SentimentIntensityAnalyzer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from keras.models import load_model

newsapi = NewsApiClient(api_key='c2dc2b55b82e43a5804b55cbcc174722')

url = 'https://newsapi.org/v2/everything'
params = {
    'q': 'yahoo finance stock market sentiment',
    'from': (datetime.datetime.now() - datetime.timedelta(days=28)).strftime('%Y-%m-%d'),
    'sortBy': 'popularity',
    'apiKey': 'c2dc2b55b82e43a5804b55cbcc174722'
}

response = requests.get(url, params=params)
data = response.json()
print(data)
articles = data['articles']

sia = SentimentIntensityAnalyzer()

sentiments = []
labels = []

for article in articles:
    title = article['title']
    description = article['description']
    text = title + ' ' + description
    sentiment = sia.polarity_scores(text)['compound']
    sentiments.append(sentiment)
    label = 1 if sentiment >= 0 else 0
    labels.append(label)

x = np.array(sentiments)
y = np.array(labels)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)



# model.save('my_model.h5')
model = load_model('my_model.h5')

loss, accuracy = model.evaluate(x_test, y_test)
print(f'Test accuracy: {accuracy:.2f}')

newsapi = NewsApiClient(api_key='c2dc2b55b82e43a5804b55cbcc174722')
new_articles = 'https://newsapi.org/v2/everything'
params = {
    'q': 'stock market',
    'from': (datetime.datetime.now() - datetime.timedelta(days=28)).strftime('%Y-%m-%d'),
    'sortBy': 'popularity',
    'apiKey': 'c2dc2b55b82e43a5804b55cbcc174722'
}

response = requests.get(new_articles, params=params)
data = response.json()
articles = data['articles']

new_sentiments = []
new_labels = []
for article in articles:
    title = article['title']
    description = article['description']
    text = title + ' ' + description
    sentiment = sia.polarity_scores(text)['compound']
    new_sentiments.append(sentiment)
    label = 1 if sentiment >= 0 else 0
    new_labels.append(label)

new_labels = model.predict(np.array(new_sentiments))
new_labels = (new_labels >= 0.5).astype(int)
# print(new_labels)
proportion_positive = sum(new_labels) / len(new_labels)
sentiment = 1 - proportion_positive
print(sentiment)

st.write(""" # Stock Market Sentiment """)
if sentiment >= 0.5:
    market_sentiment = "Positive"
else:
    market_sentiment = "Negative"
print(market_sentiment)
st.write("Market Sentiment: ", market_sentiment)
st.write("Sentiment Score: ", sentiment)

sentiment_score = np.mean(sentiments)

# Set a threshold sentiment score for predicting a stock market crash
threshold_score = -0.5

# Predict a stock market crash based on the sentiment score
if sentiment_score <= threshold_score:
    st.write('Prediction: Stock market crash likely')
else:
    st.write('Prediction: No stock market crash predicted')

# Stock Prediction Part
st.write(""" # Stock Market Prediction """)
ticker = st.text_input("Please enter stock ticker", "GOOG")

train = yf.Ticker(ticker).history(period="5y")
values = train[['Open', 'Close']].values
training_data_len = math.ceil(len(values) * 0.8)

scaler = MinMaxScaler(feature_range=(0, 1))
scaler_data = scaler.fit_transform(values)
train_data = scaler_data[0: training_data_len, :]

x_train = []
y_train_open = []
y_train_close = []

for i in range(100, len(train_data)):
    x_train.append(train_data[i - 100: i, :])
    y_train_open.append(train_data[i, 0])
    y_train_close.append(train_data[i, 1])

x_train = np.asarray(x_train)
y_train_open = np.asarray(y_train_open)
y_train_close = np.asarray(y_train_close)

seq = load_model('Open and Close Model.h5')

test_data = scaler_data[training_data_len - 100:, :]
x_test = []
y_test = values[training_data_len:, :]

for i in range(100, len(test_data)):
    x_test.append(test_data[i - 100:i, :])

x_test = np.asarray(x_test)
y_test_open = y_test[:, 0]
y_test_close = y_test[:, 1]

validation_data = scaler_data[training_data_len - 100:, :]
x_validation = []
y_validation_open = []
y_validation_close = []

predicted = seq.predict(x_test)
predicted = scaler.inverse_transform(predicted)

for i in range(100, len(validation_data)):
    x_validation.append(validation_data[i - 100:i, :])
    y_validation_open.append(validation_data[i, 0])
    y_validation_close.append(validation_data[i, 1])

x_validation = np.asarray(x_validation)
y_validation_open = np.asarray(y_validation_open)
y_validation_close = np.asarray(y_validation_close)

validation_predicted = seq.predict(x_validation)
validation_predicted = scaler.inverse_transform(validation_predicted)

validation_open = values[training_data_len:, 0]
validation_close = values[training_data_len:, 1]


st.subheader('Model Predictions for Open Price')
fig1 = pl.figure(figsize=(16, 8))
pl.ylabel('Open Price')
pl.xlabel('Date')
dates = train.iloc[training_data_len:].index
pl.plot(dates, validation_open, label='Actual')
pl.plot(dates, validation_predicted[:, 0], label='Predicted')
pl.legend()
st.pyplot(fig1)

st.subheader('Model Predictions for Close Price')
fig2 = pl.figure(figsize=(16, 8))
pl.ylabel('Close Price')
pl.xlabel('Date')
dates = train.iloc[training_data_len:].index
pl.plot(dates, validation_close, label='Actual')
pl.plot(dates, validation_predicted[:, 1], label='Predicted')
pl.legend()
st.pyplot(fig2)
