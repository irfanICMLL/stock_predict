# -*- coding: utf-8 -*-

import os
import warnings
import numpy as np
import matplotlib.pyplot as plt
import copy

import keras
from generate_series import generate_series
from keras.models import Sequential
from keras.models import load_model
from keras.layers import LSTM
from keras.layers import GRU
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Dropout
from keras.callbacks import Callback

def preprocess_data(volatility_series, bullishness_series, num_of_comments_series, time_series, time_length, train_prop, stock_price):
    """
    This function returns data for training/testing of LSTM
    :param volatility_series: 
    :param bullishness_series: 
    :param num_of_comments_series: 
    :param time_length: Time length to study
    :return: the data ready for training/testing
    """
    assert len(volatility_series) == len(time_series)
    assert len(bullishness_series) == len(time_series)
    assert len(num_of_comments_series) == len(time_series)


    preprocessed_data = []
    flag = []
    predict_days_ahead = 1
    for t in range(time_length - 1, len(volatility_series) - predict_days_ahead):
        # Append bullishness, number of comments, volatility of previous days
        temp = []
        for t_prev in range(t - time_length + 1, t + 1):
            temp.append([bullishness_series[t_prev], num_of_comments_series[t_prev], volatility_series[t_prev]])
        preprocessed_data.append(temp)

        # Generate flags for corresponding days
        # if volatility_series[t + predict_days_ahead] >= 0.5:
        #     flag.append(1)
        # else:
        #     flag.append(0)
        flag.append(volatility_series[t + predict_days_ahead])

    assert len(preprocessed_data) == len(flag)

    # Randomly permute data sequence
    # p = np.random.permutation(len(preprocessed_data))
    # preprocessed_data = np.array(preprocessed_data)[p]
    # flag = np.array(flag)[p]

    # Separate data into train and test
    train_x = preprocessed_data[:int(train_prop * len(preprocessed_data))]
    train_y = flag[:int(train_prop * len(preprocessed_data))]
    test_x = preprocessed_data[int(train_prop * len(preprocessed_data)):]
    test_y = flag[int(train_prop * len(preprocessed_data)):]
    for i in range(len(test_y)):
        if(test_y[i] >= 0.5):
            test_y[i] = 1
        else:
            test_y[i] = 0
    stock_price_test = stock_price[(len(stock_price) - len(test_x) - 1):]
    profit_without_trade = predict_profit_without_trade(stock_price_test)
    assert len(test_x) + 1 == len(stock_price_test)
    global test_set_size
    test_set_size = len(test_y)
    return (train_x, train_y), (test_x, test_y), stock_price_test, profit_without_trade

def preprocess_data_without_emotion(volatility_series, bullishness_series, num_of_comments_series, time_series, time_length, train_prop, stock_price):
    """
    This function returns data for training/testing of LSTM
    :param volatility_series: 
    :param bullishness_series: 
    :param num_of_comments_series: 
    :param time_length: Time length to study
    :return: the data ready for training/testing
    """
    assert len(volatility_series) == len(time_series)
    assert len(bullishness_series) == len(time_series)
    assert len(num_of_comments_series) == len(time_series)

    preprocessed_data = []
    flag = []
    predict_days_ahead = 1
    for t in range(time_length - 1, len(volatility_series) - predict_days_ahead):
        # Append bullishness, number of comments, volatility of previous days
        temp = []
        for t_prev in range(t - time_length + 1, t + 1):
            # temp.append([num_of_comments_series[t_prev], volatility_series[t_prev]])
            temp.append([volatility_series[t_prev]])
        preprocessed_data.append(temp)

        # Generate flags for corresponding days
        # if volatility_series[t + 1] >= 0.5:
        #     flag.append(1)
        # else:
        #     flag.append(0)
        flag.append(volatility_series[t + predict_days_ahead])

    assert len(preprocessed_data) == len(flag)

    # Randomly permute data sequence
    # p = np.random.permutation(len(preprocessed_data))
    # preprocessed_data = np.array(preprocessed_data)[p]
    # flag = np.array(flag)[p]

    # Separate data into train and test
    global test_set_size
    # train_x = preprocessed_data[:int(train_prop * len(preprocessed_data))]
    # train_y = flag[:int(train_prop * len(preprocessed_data))]
    # test_x = preprocessed_data[int(train_prop * len(preprocessed_data)):]
    # test_y = flag[int(train_prop * len(preprocessed_data)):]
    train_x = preprocessed_data[:(len(preprocessed_data) - test_set_size)]
    train_y = flag[:(len(preprocessed_data) - test_set_size)]
    test_x = preprocessed_data[(len(preprocessed_data) - test_set_size):]
    test_y = flag[(len(preprocessed_data) - test_set_size):]
    for i in range(len(test_y)):
        if(test_y[i] >= 0.5):
            test_y[i] = 1
        else:
            test_y[i] = 0
    stock_price_test = stock_price[(len(stock_price) - len(test_x) - 1):]
    profit_without_trade = predict_profit_without_trade(stock_price_test)
    assert len(test_x) + 1 == len(stock_price_test)
    print("Without emotion train length: %d" % len(train_x))
    print("Without emotion test length: %d" % len(test_x))
    return (train_x, train_y), (test_x, test_y), stock_price_test, profit_without_trade

def train(stock_name, time_length, dense_size, batch_size, epochs, train_prop, norm_window_size):
    # Read best model saved in the disk and return its accuracy
    # global best_model
    # global best_epoch
    # best_model = None
    # best_epoch = None
    def bestModelResult():
        model_path = os.path.join('data', 'results', 'models', stock_name + '.hdf5')
        model = load_model(model_path)
        flags = model.predict(test_x)
        print(flags)
        assert len(test_y) == len(flags)
        numCorrect = 0
        for (a, b) in zip(flags, test_y):
            if a >= 0.5 and b == 1 or a < 0.5 and b == 0:
                numCorrect += 1
        accuracy = float(numCorrect) / len(test_y)
        profit, mse = predict_profit(model, test_x, stock_price_test)
        profit_absolute, profit_absolute_list = predict_profit_absolute(model, test_x, stock_price_test)
        global acc_with_emotion_best_global
        global profit_absolute_with_emotion_best_global
        global stock_price_test_global
        if accuracy > acc_with_emotion_best_global:
            acc_with_emotion_best_global = accuracy
            profit_absolute_with_emotion_best_global = profit_absolute_list
            stock_price_test_global = stock_price_test
        return (accuracy, profit, profit_absolute, mse)

    # Setting up the network
    model = Sequential()
    # model.add(LSTM(32, return_sequences=True, input_shape=(time_length, 3)))
    # model.add(LSTM(dense_size))
    model.add(LSTM(dense_size, return_sequences = False, input_shape = (time_length, 3)))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(
                    # loss = 'binary_crossentropy',
                    loss = 'mean_squared_error',
                    optimizer = 'adadelta',
                    metrics = ['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.0001, patience=800, verbose=0, mode='auto')
    check_point = ModelCheckpointModified(filepath=os.path.join('data', 'results', 'models', stock_name + '.hdf5'))
    # check_point = keras.callbacks.ModelCheckpoint(filepath = os.path.join('data', 'results', 'models', stock_name + '.hdf5'),
    #                               monitor = 'val_acc',
    #                               save_best_only = True)

    time_series, volatility_series, bullishness_series, num_of_comments_series, stock_price = \
        generate_series(stock_name, norm_window_size = norm_window_size)

    volatility_series = []
    bullishness_series = []
    num_of_comments_series = []
    file_name = os.path.join('data', 'series_new', stock_name + '.txt')
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            volatility_series.append(float(line[2]))
            bullishness_series.append(float(line[1]))
            num_of_comments_series.append(float(line[0]))

    (train_x, train_y), (test_x, test_y), stock_price_test, profit_without_trade = \
        preprocess_data(volatility_series, bullishness_series,
                        num_of_comments_series, time_series, time_length, train_prop, stock_price)

    # Train model
    print("Training...")
    model.fit(train_x, train_y,
              batch_size = batch_size,
              epochs = epochs,
              validation_data = (test_x, test_y),
              callbacks = [early_stopping, check_point],
              shuffle = True)
    score, acc = model.evaluate(test_x, test_y, batch_size = batch_size)

    # profit = predict_profit(model, test_x, stock_price_test)

    print('Test score:', score)
    acc, profit, profit_absolute, mse = bestModelResult()
    print('Best Test Accuracy:', acc)

    return acc, profit, profit_absolute, profit_without_trade, mse

def train_without_emotion(stock_name, time_length, dense_size, batch_size, epochs, train_prop, norm_window_size):
    # Read best model saved in the disk and return its accuracy
    def bestModelResult():
        best_model_path = os.path.join('data', 'results', 'models without emotion', stock_name + '.hdf5')
        best_model = load_model(best_model_path)
        flags = best_model.predict(test_x)
        assert len(test_y) == len(flags)
        numCorrect = 0
        for (a, b) in zip(flags, test_y):
            if a >= 0.5 and b == 1 or a < 0.5 and b == 0:
                numCorrect += 1
        accuracy = float(numCorrect) / len(test_y)
        profit, mse = predict_profit(model, test_x, stock_price_test)
        profit_absolute, profit_absolute_list = predict_profit_absolute(model, test_x, stock_price_test)

        # Best-best model trained for this stock. save data for plotting
        global acc_without_emotion_best_global
        global profit_absolute_without_emotion_best_global
        global stock_price_test_global
        if accuracy > acc_without_emotion_best_global:
            acc_without_emotion_best_global = accuracy
            profit_absolute_without_emotion_best_global = profit_absolute_list
            stock_price_test_global = stock_price_test
        return (accuracy, profit, profit_absolute, mse)

    # Setting up the network
    model = Sequential()
    model.add(LSTM(dense_size, input_shape = (time_length, 1)))
    # model.add(LSTM(dense_size, input_shape=(time_length, 2)))
    model.add(Dense(1, activation = 'sigmoid'))
    model.compile(
                #loss = 'binary_crossentropy',
                    loss = 'mean_squared_error',
                    optimizer = 'adadelta',
                    metrics = ['accuracy'])
    early_stopping = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0, patience=800, verbose=0, mode='auto')
    check_point = ModelCheckpointModified(filepath=os.path.join('data', 'results', 'models without emotion', stock_name + '.hdf5'))
    # check_point = keras.callbacks.ModelCheckpoint(filepath=os.path.join('data', 'results', 'models without emotion', stock_name + '.hdf5'),
    #                               monitor='val_acc',
    #                               save_best_only=True)

    time_series, volatility_series, bullishness_series, num_of_comments_series, stock_price = \
        generate_series(stock_name, norm_window_size = norm_window_size)

    volatility_series = []
    bullishness_series = []
    num_of_comments_series = []
    file_name = os.path.join('data', 'series_new', stock_name + '.txt')
    with open(file_name, 'r') as f:
        for line in f.readlines():
            line = line.split('\t')
            volatility_series.append(float(line[2]))
            bullishness_series.append(float(line[1]))
            num_of_comments_series.append(float(line[0]))

    (train_x, train_y), (test_x, test_y), stock_price_test, profit_without_trade = \
        preprocess_data_without_emotion(volatility_series, bullishness_series,
                        num_of_comments_series, time_series, time_length, train_prop, stock_price)

    # Train model
    print("Training...")
    model.fit(train_x, train_y,
              batch_size = batch_size,
              epochs = epochs,
              validation_data = (test_x, test_y),
              callbacks = [early_stopping, check_point],
              shuffle = True)
    score, acc = model.evaluate(test_x, test_y, batch_size = batch_size)

    # profit = predict_profit(model, test_x, stock_price_test)

    print('Test score:', score)
    acc, profit, profit_absolute, mse = bestModelResult()
    print('Best Test Accuracy:', acc)

    return acc, profit, profit_absolute, profit_without_trade, mse

def predict_profit(model, test_x, stock_price):
    # Calculate the profit for the stock
    assert len(test_x) + 1 == len(stock_price)
    flag = model.predict(test_x)
    profit = [0.0 for i in range(len(stock_price))]
    profit[0] = 1.0
    for i in range(1, len(stock_price)):
        if flag[i - 1] >= 0.5:
            profit[i] = profit[i - 1] * stock_price[i] / stock_price[i - 1]
        else:
            profit[i] = profit[i - 1]
    print("Total Profit: %f" % profit[-1])
    # Predict the stock as a regression task. Then calculate the mse
    mse = 0
    stock_price_predict = [0.0 for i in range(len(stock_price))]
    stock_price_predict[0] = stock_price[0]
    for i in range(1, len(stock_price)):
        stock_price_predict[i] = stock_price[i - 1] * (1 + float(flag[i - 1]) / 5 - 0.1)
        mse += ((stock_price_predict[i] - stock_price[i]) / stock_price[i]) ** 2
    mse /= len(stock_price)
    return (profit[-1], mse)

def predict_profit_absolute(model, test_x, stock_price):
    assert len(test_x) + 1 == len(stock_price)
    flag = model.predict(test_x)
    profit = [0.0 for i in range(len(stock_price))]
    profit[0] = 1.0
    for i in range(1, len(stock_price)):
        fluctuation = (stock_price[i] - stock_price[i - 1]) / stock_price[i - 1]
        # Predict rise
        # if flag[i - 1] >= 0.5:
        #     if fluctuation > 0:
        #         profit *= (1 + abs(fluctuation))
        #     else:
        #         profit *= (1 - abs(fluctuation))
        # # Predict fall
        # else:
        #     if fluctuation < 0:
        #         profit *= (1 + abs(fluctuation))
        #     else:
        #         profit *= (1 - abs(fluctuation))
        if flag[i - 1] >= 0.5:
            profit[i] = profit[i - 1] * (1 + fluctuation)
        else:
            profit[i] = profit[i - 1] * (1 - fluctuation)
    return profit[-1], profit

def predict_profit_without_trade(stock_price):
    result = stock_price[-1] / stock_price[0]
    return result

class ModelCheckpointModified(Callback):
    def __init__(self, filepath, monitor = 'val_acc', verbose = 0):
        super(Callback, self).__init__()
        self.filepath = filepath
        self.monitor = monitor
        self.verbose = verbose
        self.best = -np.Inf
        self.model_saved = False
        ModelCheckpointModified.best_model = None
        ModelCheckpointModified.best_epoch = None

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        filepath = self.filepath.format(epoch=epoch, **logs)
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn('Can save best model only with %s available, '
                          'skipping.' % (self.monitor), RuntimeWarning)
        else:
            if current > self.best:
                # print('Best Model Found at Epoch %d of acc %f!!!\n\n\n\n\n\n\n\n' % (epoch, current))
                # Do not save underfit model for first n epochs
                if self.model_saved == False or epoch >= 400:
                    # global best_model
                    # global best_epoch
                    # ModelCheckpointModified.best_model = copy.deepcopy(self.model)
                    # ModelCheckpointModified.best_epoch = epoch
                    # print("Best Model of acc %f Saved\n\n\n\n\n\n\n\n\n" % current)
                    self.model.save(filepath, overwrite=True)
                    self.best = current
                    self.model_saved = True

def plot_trend(stock_name):
    global profit_absolute_with_emotion_best_global
    global profit_absolute_without_emotion_best_global
    global stock_price_test_global
    assert len(profit_absolute_with_emotion_best_global) == len(stock_price_test_global)
    assert len(profit_absolute_without_emotion_best_global) == len(stock_price_test_global)
    # Align the beginning of profit and price on plot
    ratio = stock_price_test_global[0]
    for i in range(len(stock_price_test_global)):
        profit_absolute_with_emotion_best_global[i] *= ratio
        profit_absolute_without_emotion_best_global[i] *= ratio
    # Plot
    x = np.arange(len(stock_price_test_global))
    plt.figure(figsize=(8, 4))
    plt.plot(x, stock_price_test_global, label = 'Price')
    plt.plot(x, profit_absolute_with_emotion_best_global, label = 'With Emotion')
    plt.plot(x, profit_absolute_without_emotion_best_global, label = 'Without Emotion')
    plt.xlabel('Time/Day')
    plt.ylabel('Price and Profit')
    plt.legend()
    # Save plot to disk
    file_name = os.path.join('data', 'results', stock_name, '.'.join([stock_name, 'png']))
    plt.savefig(file_name)


# Global variables
# global acc_with_emotion_best_global
# global profit_absolute_with_emotion_best_global
# global acc_without_emotion_best_global
# global profit_absolute_without_emotion_best_global
# global stock_price_test_global
acc_with_emotion_best_global = 0.0
profit_absolute_with_emotion_best_global = []
acc_without_emotion_best_global = 0.0
profit_absolute_without_emotion_best_global = []
stock_price_test_global = []
best_model = None
best_epoch = None

if __name__ == "__main__":
    stock_list_previous = [
        # '000573',
        # '000703',
        # '000733',
        # '000788',
        # '000909',
        # '300017',
        '300333',
        # '600362',
        # '601668',
        # '600605'
    ]

    stock_list = [
        # '601398',
        # '601111',
        # '601288',
        # '600690',
        # '600198',
        # '600115',
        # '600000',
        # '000043'
        # '399006'
    ]

    for stock in stock_list_previous:
        # Mean acc under different parameters
        file_name = os.path.join('data', 'results', 'parameter_sweep', 'previous', stock + '.txt')
        with open(file_name, 'w') as f:
            acc_wrt_range_with_emotion_list = []
            acc_wrt_range_without_emotion_list = []
            mse_wrt_range_with_emotion_list = []
            mse_wrt_range_without_emotion_list = []
            for time_length in range(10, 11):
                accuracy_with_emotion_list = []
                accuracy_without_emotion_list = []
                mse_with_emotion_list = []
                mse_without_emotion_list = []
                for iter_cnt in range(1):
                    print("Stock %s at length %d No.%d with emotion" % (stock, time_length, iter_cnt + 1))
                    acc_with_emotion, profit_with_emotion, profit_with_emotion_absolute, profit_without_trade, mse_with_emotion = train(
                        stock_name = stock,
                        time_length = time_length,
                        dense_size = 8,
                        batch_size = 32,
                        epochs = 5000,
                        train_prop = 0.80,
                        norm_window_size = 20
                    )
                    print("Stock %s at length %d No.%d without emotion" % (stock, time_length, iter_cnt + 1))
                    acc_without_emotion, profit_without_emotion, profit_without_emotion_absolute, profit_without_trade, mse_without_emotion = train_without_emotion(
                        stock_name = stock,
                        time_length = time_length,
                        dense_size = 4,
                        batch_size = 32,
                        epochs = 5000,
                        train_prop = 0.80,
                        norm_window_size = 20
                    )
                    accuracy_with_emotion_list.append(acc_with_emotion)
                    accuracy_without_emotion_list.append(acc_without_emotion)
                    mse_with_emotion_list.append(mse_with_emotion)
                    mse_without_emotion_list.append(mse_without_emotion)
                acc_wrt_range_with_emotion_list.append(np.mean(accuracy_with_emotion_list))
                acc_wrt_range_without_emotion_list.append(np.mean(accuracy_without_emotion_list))
                mse_wrt_range_with_emotion_list.append(np.mean(mse_with_emotion_list))
                mse_wrt_range_without_emotion_list.append(np.mean(mse_without_emotion_list))
                f.write('Time Length: ')
                f.write(str(time_length))
                f.write('\nAcc with emotion: \n')
                for num in accuracy_with_emotion_list:
                    f.write(str(num) + '\t')
                f.write('\nAcc without emotion: \n')
                for num in accuracy_without_emotion_list:
                    f.write(str(num) + '\t')
                f.write('\nMSE with emotion: \n')
                for num in mse_with_emotion_list:
                    f.write(str(num) + '\t')
                f.write('\nMSE without emotion: \n')
                for num in mse_without_emotion_list:
                    f.write(str(num) + '\t')
                f.write('\n')
            f.write('Average Accuracy With Emotion:\n')
            for num in acc_wrt_range_with_emotion_list:
                f.write(str(num) + '\t')
            f.write('\nAverage Accuracy Without Emotion:\n')
            for num in acc_wrt_range_without_emotion_list:
                f.write(str(num) + '\t')
            f.write('\nAverage MSE With Emotion:\n')
            for num in mse_wrt_range_with_emotion_list:
                f.write(str(num) + '\t')
            f.write('\nAverage MSE Without Emotion:\n')
            for num in mse_wrt_range_without_emotion_list:
                f.write(str(num) + '\t')