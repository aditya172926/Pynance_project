import bs4 as bs
import pickle
import requests
import pandas as pd
import pandas_datareader as web
import datetime as dt
import os

def sp500_tickers():
    resp = requests.get('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
    soup = bs.BeautifulSoup(resp.text, "lxml")
    table =soup.find('table', {'class': 'wikitable sortable'})
    tickers = []
    for row in table.findAll('tr')[1: ]:
        ticker = row.findAll('td')[0].text
        ticker = ticker[:-1]
        tickers.append(ticker)

    # with open("sp500tickers.pickle", "wb") as f:
    #     pickle.dump(ticker, f)
    sp500 = pd.DataFrame({'tickers': tickers})
    sp500.to_csv('sp500tickers.csv')
    
    print(tickers)
    return tickers

# sp500_tickers()

def get_stock_data():
    # got the data of 20 companies from SP 500
    start = dt.datetime(2000,1,1)
    end = dt.datetime(2016,12,31)
    sp500 = pd.read_csv('sp500tickers.csv')
    print(sp500['tickers'][:20])
    if not os.path.exists('stock_dfs'):
        os.makedirs('stock_dfs')

    for ticker in sp500['tickers'][:20]:
        if not os.path.exists('stock_dfs/{}.csv'.format(ticker)):
            df = web.DataReader(ticker, 'yahoo', start, end)
            df.to_csv('stock_dfs/{}.csv'.format(ticker))
        else:
            print('Already have {}'.format(ticker))


# get_stock_data()

# combining all the dataFrames created into 1 DF. Using only the Adj Close column

def compile_data():
    sp500 = pd.read_csv('sp500tickers.csv')
    main_df = pd.DataFrame()
    for count, ticker in enumerate(sp500['tickers'][:20]):
        df = pd.read_csv('stock_dfs/{}.csv'.format(ticker))
        df.set_index('Date', inplace=True)
        df.rename(columns = {'Adj Close': ticker}, inplace=True)
        df.drop(['Open', 'High', 'Low', 'Volume', 'Close'], axis=1, inplace=True)
        if main_df.empty:
            main_df = df
        else:
            main_df = main_df.join(df, how='outer')

        if count%10 == 0:
            print(count)
    print(main_df.head())
    main_df.to_csv('sp500_joint_adj_closes.csv')

compile_data()