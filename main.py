import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

'''
Entering in the stock of choice and getting the information for that stock
'''
chosen_stock = yf.Ticker(input('Please enter a stock ticker: '))
stock_info = chosen_stock.info
# stock_info.keys() for other properties you can explore

'''
Getting certain information of market
'''
market_price = stock_info['regularMarketPrice']
previous_close_price = stock_info['regularMarketPreviousClose']
    # day before market_price
print('market price = ', market_price)
print('previous close price = ', previous_close_price)

'''
Getting history of the stock
'''
hist = chosen_stock.history(period="max")

'''
Add a column which uses a counter instead of a date
'''

# print(type(hist))
print(hist.columns, '\n')
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
x = [i for i in range(len(train))]#hist.Counter
y = train
# #create basic scatterplot 
plt.plot(x, y, 'o') 
 
# #obtain m (slope) and b(intercept) of linear regression line 
# m, b = np.polyfit(x, y, 1) 
 
# #add linear regression line to scatterplot  
# plt.plot(x, m*x+b)		

# sns.relplot(
#     data=hist,
#     x="Date", y="Close"
# )
plt.show()

'''
Final commands
'''
# hist[close]
#print(hist[["Counter", "Open", "Close"]])
# print(hist.index)
# print(hist)

