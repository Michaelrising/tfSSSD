import datetime
import pandas as pd
import os
import yfinance as yf
import numpy as np

start = datetime.datetime(2013, 1, 2)
end = datetime.datetime(2023, 1, 1)
start_time = str(start.date())
end_time = str(end.date())

weekmasks = "Mon Tue Wed Thu Fri"
all_dates = pd.bdate_range(start=datetime.datetime(2013, 1, 1), end=datetime.datetime(2023, 1, 1), name='Date', freq='C', weekmask=weekmasks, holidays=[])
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

stocks_dic = {'DJ': DJ, 'ES': ES, 'SE': SE}

for i, index in enumerate(list(stocks_dic.keys())):
    stocks_list = stocks_dic.get(index)
    if not os.path.exists('../datasets/Stocks/' + index):
        os.makedirs('../datasets/Stocks/' + index)
    new_stock_list = []
    for ticker in stocks_list:
        file_name = '../datasets/Stocks/' + index + '/' + ticker + '_' + start_time + '_to_' + end_time + '.csv'
        print(file_name)
        # data = pdr.DataReader(ticker, 'google', start, end)
        data = yf.download(ticker, start=start_time, end=end_time)
        print(data.shape)
        if data.iloc[0].name > start:
            print(data.iloc[0].name)
            print("The history of stock {} less than 10 years, IGNORE it!".format(ticker))
            # stocks_list = stocks_list.drop(labels=i)
        else:

            index0 = all_dates.isin(data.index)
            data_w_na = pd.DataFrame(index = all_dates, columns=['Date', "Open", 'High', 'Low', 'Close', 'Adj Close', 'Volume'])
            data_w_na.iloc[index0] = data.copy(deep=True)
            data_w_na.to_csv(file_name)
            print(data_w_na.shape)
            new_stock_list.append(ticker)
    all_stocks = []
    for (i, ticker) in enumerate(new_stock_list):
        file_name = '../datasets/Stocks/' + index + '/' + ticker + '_' + start_time + '_to_' + end_time + '.csv'
        print(file_name)
        data = pd.read_csv(file_name, parse_dates=['Date'], index_col=['Date'])
        print(data.shape)
        data.to_csv(file_name)
        all_stocks.append(data.to_numpy())

    all_stocks_3dim = np.stack(all_stocks) # Num_stocks * Time_length * Features
    all_stocks_3dim = all_stocks_3dim.transpose([1, 0, 2])
    print(all_stocks_3dim.shape)
    all_stocks_file_name = '../datasets/Stocks/' + index +'_all_stocks_' + start_time + '_to_' + end_time + '.npy'
    np.save(all_stocks_file_name, all_stocks_3dim)




