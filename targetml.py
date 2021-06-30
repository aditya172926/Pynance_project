from collections import Counter
import numpy as np
import pandas as pd
from sklearn import svm, neighbors
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def process_data_for_labels(ticker):
    hm_days = 7 #this is a window of 7 days to 
    # see if we are going to buy, sell or hold a stock
    df = pd.read_csv('sp500_joint_adj_closes.csv', index_col=0)
    tickers = df.columns.values.tolist()
    df.fillna(0, inplace=True)

    for i in range(1, hm_days+1):
        df['{}_{}d'.format(ticker, i)] = (df[ticker].shift(-i) - df[ticker]) / df[ticker]
        # This makes a new column in our df, it is named as stockname_2d, and it calculates the
        # percentage change from todays price and from price after i days, that is price after i days
        # minus todays price divided by todays price.
    
    df.fillna(0, inplace=True)
    return tickers, df

def buy_sell_hold(*args):
    cols = [c for c in args]
    requirement = 0.02
    # stock price goes up by 2% in 7 days, buy
    # if goes below 2% then sell, otherwise hold
    for col in cols:
        if col > requirement:
            return 1
        if col < -requirement:
            return -1
    return 0

# now map this function our df column to generate the target column
def extract_featuresets(ticker):
    tickers, df = process_data_for_labels(ticker)
    df['{}_target'.format(ticker)] = list(map(buy_sell_hold, 
                                        df['{}_1d'.format(ticker)],
                                        df['{}_2d'.format(ticker)],
                                        df['{}_3d'.format(ticker)],
                                        df['{}_4d'.format(ticker)],
                                        df['{}_5d'.format(ticker)],
                                        df['{}_6d'.format(ticker)],
                                        df['{}_7d'.format(ticker)],
                                        ))
    
    vals = df['{}_target'.format(ticker)].values.tolist()
    str_vals = [str(i) for i in vals]
    print('Data spread:', Counter(str_vals))

    df.fillna(0, inplace=True)
    df = df.replace([np.inf, -np.inf], np.nan) #replacing the infinity values with Nan and dropping them
    df.dropna(inplace=True)

    df_vals = df[[ticker for ticker in tickers]].pct_change() #percent change wrt 1 previous day
    df_vals = df_vals.replace([np.inf, -np.inf], 0)
    df_vals.fillna(0, inplace=True)

    X = df_vals.values
    y = df['{}_target'.format(ticker)].values

    return X, y, df


def do_ml(ticker):
    X, y, df = extract_featuresets(ticker)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25)

    # clf = neighbors.KNeighborsClassifier()

    #Voting classifier
    clf = VotingClassifier([('lsvc', svm.LinearSVC()), ('knn', neighbors.KNeighborsClassifier()), ('rfor', RandomForestClassifier())])

    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    print("Accuracy", confidence)

    # the numbers may not be balanced so we have to balance the numbers and still try to stay above the 33% accuracy
    # so we can tune the classifiers or also find a sweet point between 0.02 for buy sell and hold in requirements varuable

    # To train a classifier and save it so that we won't have to train it again and again, we can pickle it.
    # So next time we just read a classifier file pickle and just run to predict the outputs
    predictions = clf.predict(X_test)

    print('Predicted spread:', Counter(predictions))

    return confidence

do_ml('MMM')
