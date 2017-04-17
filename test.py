# -*- coding: utf-8 -*-

import xlrd

readfile = xlrd.open_workbook('data/stock_comments/000573.xls')
table = readfile.sheet_by_index(0)
comments = table.col_values(2)[1:]
i = 0

for comment in comments:
    print(comment)
    i += 1
    if(i == 10):
        break
