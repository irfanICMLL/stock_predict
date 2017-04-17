# -*- coding: utf-8 -*-

"""
Prepares and returns stock series data
n stock proce returns series of length (n - 1)

Arguments:
Stock name: stock_name

Returns:
tuple(
    Time in standard format (returned in generate_volatility_series)
    Volatility (Normalized to [0, 1])
    Bullishness (Nromalized to Z-score)
    NumOfComments
)
"""

import xlrd
import os
import numpy

from datetime import datetime
from datetime import date

# from generate_volatility_series import generate_volatility_series

# Returns the volatility array of a stock

stock2tableidx = {
    '000573' : 0,
    '000703' : 1,
    '000733' : 2,
    '000788' : 3,
    '000909' : 4,
    '300333' : 5,
    '300017' : 6,
    # '600605' : 7,
    '601668' : 8,
    '600362' : 9
}

def generate_volatility_series(stock_name):
    time_series = []
    volatility_series = []

    # Read stock price
    file_name = os.path.join('data', 'stock_price.xlsx')
    readFile = xlrd.open_workbook(file_name)
    try:
        table = readFile.sheet_by_index(stock2tableidx[stock_name])
    except KeyError:
        raise Exception("Stock not found")
    stock_time = table.col_values(0)
    stock_price = [float(price) for price in table.col_values(1)]
    assert len(stock_time) == len(stock_price)

    # Generate volatility and time series for n-1 days
    price_previous_day = stock_price[0]
    for day, price in zip(stock_time[1:], stock_price[1:]):
        day = datetime(*xlrd.xldate_as_tuple(day, datemode = 0)).date()
        time_series.append(day)
        # Normalize from [-0.1, 0.1] to [0, 1]
        volatility = (price - price_previous_day) / price_previous_day
        volatility *= 5
        volatility += 0.5
        volatility_series.append(volatility)
        price_previous_day = price
    assert len(volatility_series) == len(time_series)

    return (time_series, volatility_series)


# Generate (bullishness, num of comments) for existing dates with pre-classified comment flags
def generate_bullishness_series(stock_name, time_series):
    filename = os.path.join('data', 'volatility_series', ''.join([stock_name, '.txt']))
    bullishness_series = []
    num_of_comments_series = []
    date_idx = 0

    with open(filename, 'r') as f:
        for line in f.readlines():
            date, pos_cnt, neg_cnt = line.split()
            pos_cnt = int(pos_cnt)
            neg_cnt = int(neg_cnt)
            date = datetime.strptime(date, '%Y-%m-%d').date()

            # Skip days market not open
            try:
                if date != time_series[date_idx]:
                    continue
            except IndexError:
                # Out of date range, the loop ends
                break

            bullishness = numpy.log((1 + pos_cnt) / (1 + neg_cnt))
            bullishness_series.append(bullishness)
            num_of_comments_series.append(pos_cnt + neg_cnt)
            date_idx += 1

    assert len(bullishness_series) == len(time_series)
    assert len(num_of_comments_series) == len(time_series)

    return (bullishness_series, num_of_comments_series)


def prepare_stock_data(stock_name):
    (time_series, volatility_series) = generate_volatility_series(stock_name)
    (bullishness_series, num_of_comments_series) = generate_bullishness_series(stock_name, time_series)
    return (time_series, volatility_series,
            bullishness_series, num_of_comments_series)

if __name__ == "__main__":
    stock_list = [
        '000573',
        '000703',
        '000733',
        '000788',
        '000909',
        '300333',
        '300017',
        '601668',
        '600362'
    ]

    for stock in stock_list:
        time_series, volatility_series, bullishness_series, num_of_comments_series = prepare_stock_data(stock)
        fileName = os.path.join('data', 'series', ''.join([stock, '.txt']))
        with open(fileName, 'w') as f:
            f.write('time\t\tvolatility\t\tbullishness\t\tnumber\n')
            for (time, volatility, bullishness, number) in zip(time_series, volatility_series, bullishness_series, num_of_comments_series):
                f.write(time.strftime("%Y-%m-%d"))
                f.write('\t')
                f.write(str(volatility))
                f.write('\t')
                f.write(str(bullishness))
                f.write('\t\t')
                f.write(str(number))
                f.write('\n')

    print("Generated stock data series in data/series folder")