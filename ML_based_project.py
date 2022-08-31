import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg

'''
The point of this project is to get a prediction of the market price over time of a stock, 
given the history of the stocks in the SMP 500
'''

stocks = ["TSLA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "NFLX", "NVDA"]
# for stock in stocks:
stock = "TSLA"

chosen_stock = yf.Ticker(stock)
hist = chosen_stock.history(period='2y')
hist = hist.drop(columns = ["High", "Low", "Stock Splits", "Dividends", "Volume", "Open"])
print(chosen_stock)

hist = hist.asfreq('D')
for i in hist.loc['Close']:
    if hist.loc[i][1] == 'NaN':
        hist.loc[i] = hist.loc[i-1]

hist = hist.sort_index()
hist.head()
# len_train = int(len(hist)*0.8)
# train = hist["Close"][:len_train]
# test = hist["Close"][len_train:]
# print(chosen_stock)
# print('hist length = ', len(hist))
# print('train length = ', len(train))
# print('test length = ', len(test), "\n")

steps = 55
hist_train = hist[:-steps]
hist_test  = hist[-steps:]

print(f"Train dates : {hist_train.index.min()} --- {hist_train.index.max()}  (n={len(hist_train)})")
print(f"Test dates  : {hist_test.index.min()} --- {hist_test.index.max()}  (n={len(hist_test)})")

fig, ax=plt.subplots(figsize=(9, 4))
hist_train['Close'].plot(ax=ax, label='train')
hist_test['Close'].plot(ax=ax, label='test')
ax.legend();

print(hist_train['Close'])

# y = hist_train['y']
# x = hist["Date"]
# mymodel = np.poly1d(np.polyfit(x, y, 10))
# myline = np.linspace(0, len(Counter), 100)
# plt.scatter(x, y, marker = "x", color = "green")
# Create and train forecaster
# ==============================================================================
forecaster = ForecasterAutoreg(regressor = RandomForestRegressor(random_state=123),lags = 6)
forecaster.fit(y=hist_train['Close'])
forecaster

# Predictions
# ==============================================================================
steps = 55
predictions = forecaster.predict(steps=steps)
predictions.head(5)
