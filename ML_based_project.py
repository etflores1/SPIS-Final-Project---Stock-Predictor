import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
The point of this project is to get a prediction of the market price over time of a stock, 
given the history of the stocks in the SMP 500
'''

STOCKS = ["TSLA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "NFLX", "NVDA"]
for stock in STOCKS:
    chosen_stock = yf.Ticker(stock)
    hist = chosen_stock.history(period="max")

    hist = hist.drop(columns = ["High", "Low", "Stock Splits", "Dividends", "Volume"])

    Counter = []
    for c in range(1, len(hist.index)+1):
        Counter.append(c)

    hist['Counter'] = Counter
    # hist["Counter"] = hist.index


    '''
    Define the training and testing data for the linear regression equation
    '''

    len_train = int(len(hist)*0.8)
    train = hist["Close"][:len_train]
    test = hist["Close"][len_train:]
    print('hist length = ', len(hist))
    print('train length = ', len(train))
    print('test length = ', len(test))

    '''
    Creating a graph and regression equation
    '''
    x = hist.Counter 
    # x = [i for i in range(len(train))]
    y = hist.Close 
