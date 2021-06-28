import pandas as pd
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import datetime as dt
from matplotlib import style
import plotly.graph_objs as go

style.use('ggplot')
# start = dt.datetime(2000, 1, 1)
# end = dt.datetime(2016, 1, 1)

# df = web.DataReader('TSLA', 'yahoo', start, end)
# print(df.head())

# df.to_csv('tesla.csv')

################## Got the data in a tesla.csv file #################
df = pd.read_csv('tesla.csv', parse_dates=True, index_col=0)

#getting the rolling window mean with window size as 100
# So the first 100 rows will have Nan because window 100 means, say from today and 99 previous days
# we take mean of everything. And then do the same for day 2,3,4...
# It smootherns the graph curves
df['100ma'] = df['Adj Close'].rolling(window=100, min_periods=0).mean() # min_periods actually covers the first 100 days keeping the same values as same as Adj Close
# print(df.head())

# Graphing with matplotlib
ax1 = plt.subplot2grid((6,1), (0,0), rowspan=5, colspan=1)
ax2 = plt.subplot2grid((6,1), (5,0), rowspan=1, colspan=1, sharex=ax1)

# ax1.plot(df.index, df['Adj Close'])
# ax1.plot(df.index, df['100ma'])
# ax2.bar(df.index, df['Volume'])

# plt.show()

# resampling in a dataframe means that you can resample a data within a time frame. 
# So a randomly daily collected data can be sampled to be looked lets say weeky, 
# every 10 days, monthly, yearly etc.

df_ohlc = df['Adj Close'].resample('10D').ohlc() #ohlc gives you the open high low close values. Creating a dataframe with those values
df_volume = df['Volume'].resample('10D').sum()

##################### Plotting with Plotly #############################
# We plot the data with plotly candlestick, run the code from line 46 to 53 to get the candlestick plot
# resampled data by 10 days

layout = dict(
    title = 'Tesla',
    xaxis = go.layout.XAxis(title = go.layout.xaxis.Title(text = "Time")),
    yaxis = go.layout.YAxis(title = go.layout.yaxis.Title(text = "Price US $"))
)

data = [go.Candlestick(x = df_ohlc.index,
        open = df_ohlc['open'],
        high = df_ohlc['high'],
        low = df_ohlc['low'],
        close = df_ohlc['close'])]

figSignal = go.Figure(data=data, layout=layout)
figSignal.show()

###################### Grabbing the S&P 500 data, every company in S&P 500 index fund. ##################
# getting the company names from wikipedia using beautifulsoup4. Stored in sp500tickers.pickle

