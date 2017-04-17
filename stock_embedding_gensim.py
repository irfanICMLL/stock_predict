# -*- coding: utf-8 -*-

import gensim
import xlrd
import jieba
import pickle
import os
import re
from export_embedding import export_embedding

def load_data():
    global sentences, directories

    # Load from raw data
    if not os.path.isfile('sentences.pickle'):
        print("...Loading data")
        for dir in directories:
            print("Loading file %s" % dir)
            readfile = xlrd.open_workbook(dir)
            table = readfile.sheet_by_index(0)
            titles = table.col_values(directories[dir])[1:]

            for sentence in titles:
                # sentence = str(titles[i])
                # Remove punctuation
                sentence = re.sub("[A-Za-z0-9\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）【】「」：:；;《）《》“”()\]\[»〔〕-]+", "", str(sentence))
                # Word segmentation
                sentence = jieba.lcut(sentence)
                sentences.append(sentence)

            # print("File %s Loaded Successfully" % dir)

        with open('sentences.pickle', 'wb') as handle:
            pickle.dump(sentences, handle, protocol = pickle.HIGHEST_PROTOCOL)

    # Load from disk
    else:
        with open('sentences.pickle', 'rb') as handle:
            sentences = pickle.load(handle)
        print("Data loaded from disk")

    print("%d sentences loaded" % len(sentences))

def train(embedding_dimension):
    global sentences, directories
    sentences = []
    directories = {
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/POSTS/000573.xls" : 1,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/POSTS/000703.xls" : 1,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/POSTS/000733.xls" : 1,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/POSTS/000788.xls" : 1,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/POSTS/000909.xls" : 1,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/POSTS/300017.xls" : 1,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/POSTS/300333.xls" : 1,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/POSTS/600362.xls" : 1,
        # "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/POSTS/600605.xls" : 1,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/POSTS/601668.xls" : 1,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/data/label_data.xlsx" : 1,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/所有标定原始/所有标定原始/stock2014.xlsx" : 3,
        "/Users/jonnykong/General/Activities/Lab/stockpre/sentiment_analysis/所有标定原始/所有标定原始/stock2015_cut.xlsx" : 2
    }

    load_data()

    print("...Training Model")
    model = gensim.models.Word2Vec(sentences,
                                   min_count = 20,
                                   size = embedding_dimension,
                                   workers = 4,
                                   window = 3,
                                   iter = 20)

    model.save('mymodel')
    # print("Model saved to \"mymodel\"")

    export_embedding()

if __name__ == "__main__":
    train(embedding_dimension = 20)