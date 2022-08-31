import yfinance as yf
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import datetime
from sklearn.ensemble import RandomForestRegressor
from skforecast.ForecasterAutoreg import ForecasterAutoreg
from sklearn.metrics import mean_squared_error
from statsmodels.tsa.ar_model import AutoReg

'''
The point of this project is to get a prediction of the market price over time of a stock, 
given the history of the stocks in the SMP 500
'''

stocks = ["TSLA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "NFLX", "NVDA"]
# for stock in stocks:
for stock in stocks:
    chosen_stock = yf.Ticker(stock)
    hist = chosen_stock.history(period='2y')
    hist = hist.drop(columns = ["High", "Low", "Stock Splits", "Dividends", "Volume", "Open"])
    print(chosen_stock)

    hist = hist.asfreq('D')

    hist = hist.fillna(method = 'bfill')

    # for i in hist.iterrows():
    #     if hist.loc['date']['Close'] == 'NaN':
    #         hist.loc[i+1]['Close'] = hist.loc[i]['Close']


    hist = hist.sort_index()
    hist.head()
    # len_train = int(len(hist)*0.8)
    # train = hist["Close"][:len_train]
    # test = hist["Close"][len_train:]
    # print(chosen_stock)
    # print('hist length = ', len(hist))
    # print('train length = ', len(train))
    # print('test length = ', len(test), "\n")

    steps = 7
    hist_train = hist[:-steps]
    hist_test  = hist[-steps:]

    print(f"Train dates : {hist_train.index.min()} --- {hist_train.index.max()}  (n={len(hist_train)})")
    print(f"Test dates  : {hist_test.index.min()} --- {hist_test.index.max()}  (n={len(hist_test)})")

    print(hist_train['Close'])

    # y = hist_train['y']
    # x = hist["Date"]
    # mymodel = np.poly1d(np.polyfit(x, y, 10))
    # myline = np.linspace(0, len(Counter), 100)
    # plt.scatter(x, y, marker = "x", color = "green")
    print(hist_train)
    # Create and train forecaster
    forecaster = ForecasterAutoreg(regressor = RandomForestRegressor(random_state=123),lags = 6)
    forecaster.fit(y=hist_train['Close'])
    forecaster

    # Predictions
    steps = 7
    predictions = forecaster.predict(steps=steps)
    predictions.head(5)

    error = mean_squared_error(y_true = hist_test['Close'], y_pred = predictions)
    print(f"Test error = ", error)

    regressor = RandomForestRegressor(max_depth=6, n_estimators=500, random_state=123)
    forecaster = ForecasterAutoreg(regressor = regressor, lags = 20)
    forecaster.fit(y=hist_train['Close'])


    fig, ax=plt.subplots(figsize=(9, 4))
    hist_train['Close'].plot(ax=ax, label='train')
    hist_test['Close'].plot(ax=ax, label='test')
    predictions.plot(ax=ax, label='predictions')
    ax.legend(),

    # Making plot and organizing it
    plt.title('Stock Prediction Model')
    plt.xlabel('Date')
    plt.ylabel('Stock Price ($)')
    plt.xlim(datetime.date(2022, 7, 29), datetime.date(2022, 8, 30))
    plt.show()

    