# -*- coding: utf-8 -*-

from keras.models import load_model
from datetime import date
import xlrd
import re
import codecs
import jieba
import numpy as np
import os

def generate_posneg_series(comments_file, outfile, max_len, embedding_dimension):
    model = load_model('sentiment_classification_model.h5')
    readfile = xlrd.open_workbook(comments_file)
    table = readfile.sheet_by_index(0)
    titles = table.col_values(1)[1:]
    comments = table.col_values(2)[1:]
    dates = table.col_values(3)[1:]


    # Stopwords
    stopwords_set = set()
    f = codecs.open('data/stopwords.txt', 'r', encoding = 'gb18030')
    while True:
        line = f.readline()
        if line:
            stopwords_set.add(line.rstrip('\n'))
        else:
            break
    f.close()

    # Construct word to embedding dict with embedding.txt
    word2embedding = dict()
    with open(os.path.join('data', 'embeddings.txt'), 'r') as f:
        for lines in f.readlines():
            word, embedding = lines.split('\t')
            word2embedding[word] = [float(i) for i in embedding.split()]

    # Construct word to sentiment dict with words_utf8.txt
    word2sentiment = dict()
    with open(os.path.join('data', 'words_utf8.txt')) as f:
        for line in f.readlines():
            try:
                word, sentiment = line.split()
            except ValueError:
                # Corrupt Data
                continue
            word2sentiment[word] = float(sentiment)

    # Modify word to embedding dict
    for word in word2embedding:
        if word in word2sentiment:
            word2embedding[word].append(word2sentiment[word])
        else:
            word2embedding[word].append(0.)

    # Iterate through all comments
    preprocessed_data = []
    date_of_comments = []
    for (title, comment, day) in zip(titles, comments, dates):
        # Use title if comments are empty
        if len(comment) == 0:
            comment = title
        # Skip if both title and comment are empty
        if len(comment) == 0:
            continue

        day = day.split()[0]
        day = day.split(sep = '-')
        day = date(int(day[0]), int(day[1]), int(day[2]))

        # Append comments to check into list
        # if check_cnt == 0:
        #     check_comments.append(comment)
        #     check_cnt = 100
        # check_cnt -= 1

        comment = re.sub("[A-Za-z0-9\s+\.\!\/_,$%^*(+\"\']+|[+——！，。？、~@#￥%……&*（）【】「」：:；;《）《》“”()\]\[»〔〕-]+", "", str(comment))
        comment = jieba.lcut(comment)
        comment = [c for c in comment if c not in stopwords_set]

        temp = []
        # i = 0
        # for word in comment:
        #     i += 1
        #     if word in word2embedding:
        #         temp.append(word2embedding[word])
        #     else:
        #         i -= 1   # Skip over
        #     if i >= max_len:
        #         break
        # while i < max_len:
        #     i += 1
        #     temp.append([0 for j in range(embedding_dimension)])
        i = 0
        for word in comment:
            if word in word2embedding:
                temp.append(word2embedding[word])
                i += 1
            else:
                continue
            if i >= max_len:
                break
        while i < max_len:
            i += 1
            temp.append([0 for j in range(embedding_dimension + 1)])
        preprocessed_data.append(temp)
        date_of_comments.append(day)

    preprocessed_data = np.array(preprocessed_data)
    flags = model.predict(preprocessed_data)
    assert len(preprocessed_data) == len(flags)
    for i in range(len(flags)):
        if flags[i] >= 0.5:
            flags[i] = int(1)
        else:
            flags[i] = int(0)

    # print("Percentage of Pos: %f" % (sum(flags) / len(flags)))

    # Records the number of pos and neg comments on a specific date
    date2sentimentCount = dict()
    for (flag, day) in zip(flags, date_of_comments):
        if day not in date2sentimentCount:
            date2sentimentCount[day] = [0, 0] # POS, NEG
        if flag == 1:
            date2sentimentCount[day][0] += 1
        else:
            date2sentimentCount[day][1] += 1

    # for day in date2sentimentCount:
    #     print(day, end = ' ')
    #     print(date2sentimentCount[day][0], end = ' ')
    #     print(date2sentimentCount[day][1])

    with open(outfile, 'w') as f:
        s = sorted(date2sentimentCount.items(), key = lambda x : x[0])
        for (day, num) in s:
            f.write(str(day))
            f.write('\t')
            f.write(str(num[0]))
            f.write('\t')
            f.write(str(num[1]))
            f.write('\n')

    # TODO: Display checked comments
    # for i in range(len(check_comments)):
    #     print(flags[100 * i], end = '\t')
    #     # print(check_comments[i])
    #     if len(check_comments[i]) < 30:
    #         print(check_comments[i])
    #     else:
    #         print(check_comments[i][0:50])

    # print("Total number of comments: %d" % len(preprocessed_data))
    print("Successfully generated sentiment series for %s" % comments_file)

if __name__ == "__main__":
    # generate_series('data/stock_comments/000573.xls',
    #                 'data/stock_price.xls',
    #                 'data/votality_series/000573.txt',
    #                 max_len = 30,
    #                 embedding_dimension = 20)
    comments_list = [
                     # '000573.xls',
                     # '000703.xls',
                     # '000733.xls',
                     # '000788.xls',
                     # '000909.xls',
                     # '300017.xls',
                     # '300333.xls',
                     # '600362.xls',
                     # '601668.xls',
                     # '601398.xls',
                     # '601111.xls',
                     # '600115.xls',
                     # '600000.xls',
                     # '600198.xls',
                     # '600690.xls',
                     # '601288.xls',
                     # '601288_2.xls',
                     # '000043.xls',
                     '600605.xls'
                    ]
    results_list = [
                    # '000573.txt',
                    # '000703.txt',
                    # '000733.txt',
                    # '000788.txt',
                    # '000909.txt',
                    # '300017.txt',
                    # '300333.txt',
                    # '600362.txt',
                    # '601668.txt',
                    # '601398.txt',
                    # '601111.txt',
                    # '600115.txt',
                    # '600000.txt',
                    # '600198.txt',
                    # '600690.txt',
                    # '601288.txt',
                    # '601288_2.txt',
                    # '000043.txt',
                    '600605.txt'
                    ]
    for(comment_dir, result_dir) in zip(comments_list, results_list):
        generate_posneg_series(os.path.join('data/stock_comments', comment_dir),
                               os.path.join('data/posneg_series', result_dir),
                               max_len = 30,
                               embedding_dimension = 20)