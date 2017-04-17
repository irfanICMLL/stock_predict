# -*- coding: utf-8 -*-

import jieba
import xlrd

import random
import string, re
import codecs
import os

import numpy as np
import theano
import theano.tensor as T
import gensim

from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense

# Load data from xlsx into data
# Conduct punctuation & digits & alphabets removal
# Conduct word segmentation
def load_data(stopwords = 'stopwords.txt'):
    print("...Loading Data", end = '')
    data = []

    # Read stock2015_cut.xlsx
    readFile = xlrd.open_workbook('data/stock2015_cut.xlsx')
    table = readFile.sheet_by_index(0)
    titles = table.col_values(2)[1:]
    comments = table.col_values(3)[1:]
    sentiments = table.col_values(4)[1:]

    # Read pos
    readFile = xlrd.open_workbook('data/pos.xlsx')
    table = readFile.sheet_by_index(0)
    comments += table.col_values(0)[0:]
    titles += table.col_values(0)[0:]
    sentiments += ['1' for i in range(len(table.col_values(0)[0:]))]

    # Read neg
    readFile = xlrd.open_workbook('data/neg.xlsx')
    table = readFile.sheet_by_index(0)
    comments += table.col_values(0)[0:]
    titles += table.col_values(0)[0:]
    sentiments += ['-1' for i in range(len(table.col_values(0)[0:]))]

    # Stopwords
    stopwords_set = set()
    stopwords_path = os.path.join('data', stopwords)
    f = codecs.open(stopwords_path, 'r', encoding = 'gb18030')
    while True:
        line = f.readline()
        if line:
            stopwords_set.add(line.rstrip('\n'))
        else:
            break
    f.close()

    i = 0
    while i < len(sentiments):
        if sentiments[i] == '':
            i += 1
            continue
        if int(sentiments[i]) == 0:
            i += 1
            continue
        if True:
            # If comment is empty, use its title instead
            comment_input = ''
            if comments[i] != '':
                comment_input = str(comments[i])
            else:
                comment_input = str(titles[i])
            # Remove superfluous chars with regular expression
            comment_input = re.sub("[A-Za-z0-9\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）【】「」：:；;《）《》“”()\]\[»〔〕-]+", "", comment_input)
            # Conduct word segmentation
            comment_input = jieba.lcut(comment_input)
            # Truncate stopwords
            comment_input = [comment for comment in comment_input if comment not in stopwords_set]
            # Append to data list
            # if int(sentiments[i]) == 1:
            #     data.append((comment_input, 1))
            # else:
            #     data.append((comment_input, 0))
            if len(comment_input) > 0:
                data.append((comment_input, np.floor(float(sentiments[i]) + 0.1)))
            i += 1


    random.shuffle(data)
    print("...Done!")
    return data

def preprocess_data(data, max_len, embedding_dimension, train_prop):
    '''
    :param data: raw data
    :return: numpy array of train/test data
    '''

    # max_len = 20    # Concatenate or pad zeros if not satisfied
    # train_prop = 0.8

    # Create word2idx
    word2embedding = dict()
    with open("data/embeddings.txt", 'r') as f:
        for line in f.readlines():
            word, embedding = line.split('\t')
            word2embedding[word] = [float(i) for i in embedding.split()]
            # assert len(word2embedding[word]) == embedding_dimension

    # TODO: Create word2sentiment
    word2sentiment = dict()
    with open(os.path.join('data', 'words_utf8.txt')) as f:
        for line in f.readlines():
            try:
                word, sentiment = line.split()
            except ValueError:
                # Corrupt Data
                continue
            word2sentiment[word] = float(sentiment)
    # TODO: Modify word2embedding
    for word in word2embedding:
        if word in word2sentiment:
            word2embedding[word].append(word2sentiment[word])
        else:
            word2embedding[word].append(0)

    preprocessed_data = []
    for comment_flag in data:
        comment = comment_flag[0]
        temp = []
        # i = 0
        # for word in comment:
        #     i += 1
        #     if word in word2embedding:
        #         temp.append(word2embedding[word])
        #     else:
        #         # Pad zeros if word not found
        #         # temp.append([0 for j in range(embedding_dimension)])
        #         # Skip over if not found
        #         i -= 1
        #     if i >= max_len:
        #         break
        # while i < max_len:
        #     i += 1
        #     temp.append([0 for j in range(embedding_dimension)])
        # assert len(temp) == max_len
        # assert len(temp[0]) == embedding_dimension
        # preprocessed_data.append(temp)

        # TODO: Added sentimental dimension to the end
        i = 0
        for word in comment:
            if word in word2embedding:
                temp.append(word2embedding[word])
                i += 1
            else:
                continue  # Pad zeros
            if i >= max_len:
                break
        while i < max_len:
            i += 1
            temp.append([0 for j in range(embedding_dimension + 1)])
        assert len(temp) == max_len
        assert len(temp[0]) == embedding_dimension + 1
        preprocessed_data.append(temp)

    flag = []
    for i in range(len(data)):
        if data[i][1] == 1:
            flag.append(1)
        else:
            flag.append(0)
    # assert len(flag) == len(preprocessed_data)

    train_x = np.array(preprocessed_data[:int(train_prop * len(data))])
    train_y = np.array(flag[:int(train_prop * len(data))])
    test_x = np.array(preprocessed_data[int(train_prop * len(data)):])
    test_y = np.array(flag[int(train_prop * len(data)):])

    return (train_x, train_y), (test_x, test_y)


def train(max_len, embedding_dimension, dense_size, batch_size, epochs, train_prop):

    # Load and prepare data
    data = load_data()
    (train_x, train_y), (test_x, test_y) = preprocess_data(data, max_len, embedding_dimension, train_prop)
    # assert len(train_x) == len(train_y)
    # assert len(test_x) == len(test_y)
    # assert len(train_x[0]) == max_len and len(test_x[0]) == max_len
    # assert len(train_x[0][0]) == 10 and len(test_x[0][0]) == 10

    # Setting up network
    model = Sequential()
    # model.add(LSTM(dense_size, input_shape = (max_len, embedding_dimension), dropout = 0.2))
    # TODO: Added sentimental dimension to the end
    model.add(LSTM(dense_size, input_shape=(max_len, embedding_dimension + 1), dropout=0.2))
    model.add(Dense(1, activation = 'sigmoid'))

    model.compile(loss = 'binary_crossentropy',
                  optimizer = 'rmsprop',
                  metrics = ['accuracy'])

    print("Training...")
    model.fit(train_x, train_y,
              batch_size = batch_size,
              epochs = epochs,
              validation_data = (test_x, test_y))
    score, acc = model.evaluate(test_x, test_y, batch_size = batch_size)

    print('Test score:', score)
    print('Test accuracy:', acc)

    model.save('sentiment_classification_model.h5')
    print('Model saved to "sentiment_classification_model.h5"')

if __name__ == "__main__":
    train(max_len = 30,
          embedding_dimension = 20,
          dense_size = 64,
          batch_size = 32,
          epochs = 30,
          train_prop = 0.9)