import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
The point of this project is to get a prediction of the market price over time of a stock, 
given the history of the stocks in the SMP 500
'''

stocks = ["TSLA", "MSFT", "AAPL", "AMZN", "GOOGL", "META", "NFLX", "NVDA"]
for stock in stocks:
    chosen_stock = yf.Ticker(stock)
    hist = chosen_stock.history(period="max")
    hist = hist.drop(columns = ["High", "Low", "Stock Splits", "Dividends", "Volume", "Open"])
    print(chosen_stock, len(hist))
