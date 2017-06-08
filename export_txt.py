# -*- coding: utf-8 -*-

import os
import xlrd

# infile_name = '/Users/jonnykong/Desktop/stock2015_cut.xlsx'
# outfile_name = '/Users/jonnykong/Desktop/stock2015_cut.txt'
# infile_name = '/Users/jonnykong/Desktop/label_data.xlsx'
# outfile_name = '/Users/jonnykong/Desktop/label_data.txt'
infile_name = '/Users/jonnykong/Desktop/neg.xlsx'
outfile_name = '/Users/jonnykong/Desktop/neg.txt'

if __name__ == "__main__":
    readFile = xlrd.open_workbook(infile_name)
    table = readFile.sheet_by_index(0)
    titles = table.col_values(0)[1:]
    # comments = table.col_values(3)[1:]
    # flags = table.col_values(4)[1:]

    i = 0
    with open(outfile_name, 'w') as f:
        # for (title, comment, flag) in zip(titles, comments, flags):
        for title in titles:
        # for (title, flag) in zip(titles, flags):
            f.write(str(title))
            f.write('\t')
            f.write('-1')
            # f.write(str(comment))
            # if flag == 1 or str(flag) == '1':
            #     f.write('\t')
            #     f.write('1')
            #     i += 1
            # elif flag == 0 or str(flag) == '0':
            #     f.write('\t')
            #     f.write('0')
            #     i += 1
            # elif flag == -1 or str(flag) == '-1':
            #     f.write('\t')
            #     f.write('-1')
            #     i += 1
            f.write('\n')
