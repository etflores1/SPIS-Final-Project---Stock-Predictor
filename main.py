'''
Imports, Data Sets, Libraries
'''
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

'''
Entering in the stock of choice and getting the information for that stock
'''
print(' ')
chosen_stock = yf.Ticker(input('Please enter a stock ticker: '))
print(' ')
stock_info = chosen_stock.info
# stock_info.keys() for other properties you can explore

'''
Getting certain information of market
'''
market_price = stock_info['regularMarketPrice']
previous_close_price = stock_info['regularMarketPreviousClose']
    # day before market_price
print('Current market price = ', market_price)
print(' ')
# print('previous close price = ', previous_close_price)

'''
Getting history of the stock
'''
hist = chosen_stock.history(period='max')
    # time_lengths = ['1d','5d','1mo','3mo','6mo','1y','2y','5y','10y','max']

'''
Only use data from close market prices
Add a column which uses a counter instead of a date
'''
# print(hist.columns, ' ')
hist = hist.drop(columns = ["High", "Low", "Stock Splits", "Dividends", "Volume"])

Counter = []
for c in range(1, len(hist.index)+1):
    Counter.append(c)

hist['Counter'] = Counter

'''
Define the training and testing data for the linear regression equation
'''

len_train = int(len(hist)*0.8)
train = hist["Close"][:len_train]
test = hist["Close"][len_train:]
# print('hist length = ', len(hist))
# print('train length = ', len(train))
# print('test length = ', len(test))

'''
Creating a graph and regression equation
'''
x = hist.Counter 
# x = [i for i in range(len(train))]
y = hist.Close 
# y = train
# # #create basic scatterplot 
# plt.plot(x, y, 'o') 
 
# # #obtain m (slope) and b(intercept) of linear regression line 
# m, b = np.polyfit(x, y, 1) 
# b = 0
# print('y', '=', m,'x')


# # #add linear regression line to scatterplot  
# plt.plot(x, m*x+b)		

# # sns.relplot(
# #     data=hist,
# #     x="Date", y="Close"
# # )

'''
Polynomial Regression equation
'''
equation = np.poly1d(np.polyfit(x, y, 10))
# print(equation)

'''
Creating Scatter Plot and Regression Line
'''
mymodel = np.poly1d(np.polyfit(x, y, 10))
myline = np.linspace(0, len(Counter), 100)
plt.scatter(x, y, marker = "x", color = "green")
line, = plt.plot(myline, mymodel(myline))
line.set_color("red")

'''
Finding the market value of a date in the future
'''
predicted_market_date = int(input('Enter number of days in the future: '))
print(' ')

x = predicted_market_date + len(hist)
y = np.polyval(equation,x)

print('predicted market value of', chosen_stock, '=', y)
print(' ')


'''
Graphing of the plot
'''
plt.title('Stock Prediction Model')
plt.xlabel('Days from IPO')
plt.ylabel('Close Price ($)')
plt.show()