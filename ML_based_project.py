import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
The point of this project is to get a prediction of the market price over time of a stock, 
given the history of the stocks in the SMP 500
'''

SMP_500 = []
for yf.ticker() in 'SPY':
    SMP_500
yf.ticker('SPY')
print(SMP_500)

