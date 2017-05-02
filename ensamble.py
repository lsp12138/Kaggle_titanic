#!/usr/bin/env python
# encoding: utf-8

"""
@version: 
@author: 
@time: 2016/12/25 14:22
@remark:
"""
import pandas as pd

def ensamble():
    array1 = pd.read_csv(r'E:\kaggle\titanicv2\sub\sub1.csv', header=None).values
    array2 = pd.read_csv(r'E:\kaggle\titanicv2\sub\sub2.csv', header=None).values
    array3 = pd.read_csv(r'E:\kaggle\titanicv2\sub\sub3.csv', header=None).values
    array4 = pd.read_csv(r'E:\kaggle\titanicv2\sub\sub4.csv', header=None).values
    array5 = pd.read_csv(r'E:\kaggle\titanicv2\sub\sub5.csv', header=None).values
    array6 = pd.read_csv(r'E:\kaggle\titanicv2\sub\sub6.csv', header=None).values
    array7 = pd.read_csv(r'E:\kaggle\titanicv2\sub\sub7.csv', header=None).values
    newarray = pd.read_csv(r'E:\kaggle\titanicv2\sub\sub5.csv', header=None).values


    for i in range(1,len(array1)):
        count = 0
        if (array1[i][1] == str(1)): count = count + 1
        if (array2[i][1] == str(1)): count = count + 1
        if (array3[i][1] == str(1)): count = count + 1
        if (array4[i][1] == str(1)): count = count + 1
        if (array5[i][1] == str(1)): count = count + 1
        if (array6[i][1] == str(1)): count = count + 1
        if (array7[i][1] == str(1)): count = count + 1

        print count
        if (count > 3): newarray[i][1] = 1
        else: newarray[i][1] = 0
    df = pd.DataFrame(newarray,index=None)
    df.to_csv(r'E:\kaggle\titanicv2\sub\finalsub.csv',index=None,header=None)


    # print array2[0],array2[1],array2[1][0],type(array2[1][1])
    # ['PassengerId' 'Survived'] ['892' '0'] 这个array2[1][1]读出来是str格式，我在上边用int比一个都没比出来，要转下型


class Main():
    def __init__(self):
        pass


if __name__ == '__main__':
    ensamble()