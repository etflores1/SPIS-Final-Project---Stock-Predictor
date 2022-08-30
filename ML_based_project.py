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
for stock in stocks:
    chosen_stock = yf.Ticker(stock)
    hist = chosen_stock.history(period='2y')
    hist = hist.drop(columns = ["High", "Low", "Stock Splits", "Dividends", "Volume", "Open"])
    print(chosen_stock)
    # len_train = int(len(hist)*0.8)
    # train = hist["Close"][:len_train]
    # test = hist["Close"][len_train:]
    # print(chosen_stock)
    # print('hist length = ', len(hist))
    # print('train length = ', len(train))
    # print('test length = ', len(test), "\n")

    steps = 450
    data_train = hist['Close'][:steps]
    data_test  = hist['Close'][steps:]

    print(f"Train dates : {data_train.index.min()} --- {data_train.index.max()}  (n={len(data_train)})")
    print(f"Test dates  : {data_test.index.min()} --- {data_test.index.max()}  (n={len(data_test)})")

    fig, ax=plt.subplots(figsize=(9, 4))
    data_train.plot(ax=ax, label='train')
    data_test.plot(ax=ax, label='test')
    ax.legend();

    # Create and train forecaster
    # ==============================================================================
    forecaster = ForecasterAutoreg(regressor = RandomForestRegressor(random_state=123),lags = 6)
    forecaster.fit(y=data_train)
    forecaster

    # Predictions
    # ==============================================================================
    steps = 54
    predictions = forecaster.predict(steps=steps)
    predictions.head(5)
