import numpy as np
import csv

class myCSV():
    def open(name, ct=-1, hasHead=True, delimiter=','):
        idx, data, dic, c = [], [], {}, 0
        with open(name, newline='') as csvfile:
            spamreader = csv.reader(csvfile, delimiter=delimiter, quotechar='|')
            if ct < 0:
                ct = len(spamreader)
            for row in spamreader:
                if c == 0 and hasHead == True:
                    idx = row
                    c -= 1
                else:
                    data.append(row)
                c+=1
                if c >= ct:
                    break
        if hasHead == True:
            for i in range(len(idx)):
                dic[idx[i]] = i
        return dic, data

    def select(dic, data, val, typTrans):
        t = []
        for i in data:
            tt = []
            for j in range(len(val)):
                v = i[dic[val[j]]]
                v = typTrans(v)
                tt.append(v)
            t.append(tt)
        return t