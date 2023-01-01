import pandas_datareader as pdr
import datetime
import pandas as pd

start = datetime.datetime(2013, 1, 1)
end = datetime.datetime(2023, 1, 1)
start_date_str = str(start.date())
end_date_str = str(end.date())

DJ_stocks = ['MMM', 'AXP', 'AAPL', 'BA', 'CAT', 'CVX', 'CSCO', 'KO', 'DIS', 'XOM', 'GE',
          'GS', 'HD', 'IBM', 'INTC', 'JNJ', 'JPM', 'MCD', 'MRK', 'MSFT', 'NKE', 'PFE',
          'PG', 'TRV', 'UTX', 'UNH', 'VZ', 'WMT', 'GOOGL', 'AMZN', 'AABA']

ES_stocks =

def get_data(stocks_list, start_time, end_time):
    for ticker in stocks_list:
        file_name = 'datasets/' + str() + ticker + '_' + start_date_str + '_to_' + end_date_str + '.csv'
        print(file_name)
        data = pdr.DataReader(ticker, 'google', start, end)
        print(data.shape)
        data.to_csv(file_name)

    for (i, ticker) in enumerate(DJ_stocks):
        file_name = 'datasets/DJ_stocks/' + ticker + '_' + start_date_str + '_to_' + end_date_str + '.csv'
        print(file_name)
        data = pd.read_csv(file_name, parse_dates=['Date'], index_col=['Date'])
        print(data.shape)
        data['Name'] = ticker
        data.to_csv(file_name)

        if i == 0:
            all_stocks = data
        else:
            all_stocks = all_stocks.append(data)

    print(all_stocks.shape)
    all_stocks_file_name = 'data/all_stocks_' + start_date_str + '_to_' + end_date_str + '.csv'
    all_stocks.to_csv(all_stocks_file_name)
