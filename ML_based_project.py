'''
Imports, Data Sets, Libraries
'''
import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg

'''
The point of this project is to get a prediction of the market price over time of a stock, 
given the history of the stocks in the list
'''

'''
We are looping so we can get a prediction for all the stocks in the list
'''
stocks = ["TSLA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "NFLX", "NVDA"]
for stock in stocks:
    chosen_stock = yf.Ticker(stock)
    hist = chosen_stock.history(period='2y')
    hist = hist.drop(columns = ["High", "Low", "Stock Splits", "Dividends", "Volume", "Open"])
    print('\n', chosen_stock, '\n')

    '''
    Data Preparation
    '''
    hist = hist.asfreq('D')
        # Frequency = days
    hist = hist.fillna(method = 'bfill')
        # Every day that doesn't have a closing price will assume the closing price value of the day after it
    hist = hist.sort_index()
    hist.head()

    '''
    Spliting data into trained data and test data (to see how accurate the prediction is)
    '''
    steps = 7
    hist_train = hist[:-steps]
    hist_test  = hist[-steps:]

    print(f"Train dates : {hist_train.index.min()} --- {hist_train.index.max()}  (n={len(hist_train)})")
    print(f"Test dates  : {hist_test.index.min()} --- {hist_test.index.max()}  (n={len(hist_test)})")
    print(' ')

    '''
    Create and train forecaster
    '''
    forecaster = ForecasterAutoreg(regressor = RandomForestRegressor(random_state=123),lags = 6)
    forecaster.fit(y=hist_train['Close'])
    forecaster

    # Predictions
    steps = 7
    predictions = forecaster.predict(steps=steps)
    predictions.head(5)

    regressor = RandomForestRegressor(max_depth=6, n_estimators=500, random_state=123)
    forecaster = ForecasterAutoreg(regressor = regressor, lags = 20)
    forecaster.fit(y=hist_train['Close'])

    '''
    Finding how erroneous the prediction is
    '''
    error = mean_squared_error(y_true = hist_test['Close'], y_pred = predictions)
    print(f"Test error = ", error,'\n')

    '''
    Plotting and organizing the training data points, testing data points, and prediction graph into a legend
    '''
    fig, ax=plt.subplots(figsize=(9, 4))
    hist_train['Close'].plot(ax=ax, label='train')
    hist_test['Close'].plot(ax=ax, label='test')
    predictions.plot(ax=ax, label='predictions')
    ax.legend(),

    plt.title('Stock Prediction Model: ' + stock)
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.xlim(datetime.date(2022, 7, 29), datetime.date(2022, 8, 30))
    plt.show()

    