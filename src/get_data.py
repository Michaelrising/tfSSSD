import datetime
import pandas as pd
import os
import yfinance as yf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

start = datetime.datetime(2018, 1, 2)
end = datetime.datetime(2023, 1, 1)
start_time = str(start.date())
end_time = str(end.date())

weekmasks = "Mon Tue Wed Thu Fri"
all_dates = pd.bdate_range(start=datetime.datetime(2018, 1, 1), end=datetime.datetime(2023, 1, 1), name='Date', freq='C', weekmask=weekmasks, holidays=[])
# all_dates = all_dates.dt.dayofweek
url0 = 'https://en.wikipedia.org/wiki/Dow_Jones_Industrial_Average'
tb0 = pd.read_html(url0)[1]
DJ = tb0.Symbol
url = 'https://en.wikipedia.org/wiki/EURO_STOXX_50'
tb = pd.read_html(url)[3]
ES = tb.Ticker
url1 = 'https://en.wikipedia.org/wiki/Hang_Seng_Index'
tb1 = pd.read_html(url1)[6]
SE = tb1.Ticker.apply(lambda ticker: ticker.replace('SEHK:\xa0', ''))
SE = SE.apply(lambda ticker: ticker.zfill(4) + '.HK')

url2 = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies#S&P_500_component_stocks'
tb2 = pd.read_html(url2)[0]
SP500 = tb2.Symbol

tb3 = pd.read_csv('../datasets/Stocks/HSCI.csv')
HSCI = tb3['Code'].apply(lambda ticker: str(ticker).zfill(4) + '.HK')

tb4 = pd.read_csv('../datasets/Stocks/SP350EU.csv')
SP350 = tb4['Ticker']

US = pd.unique(pd.concat((DJ, SP500)))
HK = pd.unique(pd.concat((SE, HSCI)))
EU = pd.unique(pd.concat((ES, SP350)))
stocks_dic = {'US': US, "HK":HK, 'EU': EU} #

for i, index in enumerate(list(stocks_dic.keys())):
    stocks_list = stocks_dic.get(index)
    if not os.path.exists('../datasets/Stocks/' + index):
        os.makedirs('../datasets/Stocks/' + index)
    new_stock_list = []
    all_stocks = []
    for ticker in stocks_list:
        file_name = '../datasets/Stocks/' + index + '/' + ticker + '_' + start_time + '_to_' + end_time + '.csv'
        scaled_file_name = '../datasets/Stocks/' + index + '/scaled_' + ticker + '_' + start_time + '_to_' + end_time + '.csv'
        print(file_name)
        # data = pdr.DataReader(ticker, 'google', start, end)
        data = yf.download(ticker, start=start_time, end=end_time)
        print(data.shape)
        if not data.empty:
            if data.iloc[0].name > start:
                print(data.iloc[0].name)
                print("The history of stock {} less than 5 years, IGNORE it!".format(ticker))
                # stocks_list = stocks_list.drop(labels=i)
            else:
                index0 = all_dates.isin(data.index)
                data_w_na = pd.DataFrame(index = all_dates, columns=["Open", 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
                data_w_na.iloc[index0] = data.copy(deep=True)
                data_w_na.to_csv(file_name)
                scalar0 = MinMaxScaler()
                scalar1 = MinMaxScaler()
                price = scalar0.fit_transform(data[["Open", 'High', 'Low', 'Close', 'Adj Close']])
                volume = scalar1.fit_transform(data['Volume'].to_numpy().reshape(-1,1))
                scaled_data = np.concatenate((price, volume), axis=1)
                scaled_data_w_na = pd.DataFrame(index=all_dates,
                                         columns=["Open", 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
                scaled_data_w_na.iloc[index0] = scaled_data.copy()
                scaled_data_w_na.to_csv(scaled_file_name)
                print(data_w_na.shape)
                new_stock_list.append(ticker)
                all_stocks.append(scaled_data_w_na.to_numpy())

    # for (i, ticker) in enumerate(new_stock_list):
    #     file_name = '../datasets/Stocks/' + index + '/scaled_' + ticker + '_' + start_time + '_to_' + end_time + '.csv'
    #     print(file_name)
    #     data = pd.read_csv(file_name, parse_dates=['Date'], index_col=['Date'])
    #     print(data.shape)
    #     data.to_csv(file_name)
    #     all_stocks.append(data.to_numpy())

    all_stocks_3dim = np.stack(all_stocks, axis=0) # Num_stocks * Time_length * Features
    all_stocks_3dim = all_stocks_3dim.transpose([1, 0, 2])
    print(all_stocks_3dim.shape)
    all_stocks_file_name = '../datasets/Stocks/scaled_' + index +'_all_stocks_' + start_time + '_to_' + end_time + '.npy'
    np.save(all_stocks_file_name, all_stocks_3dim)




