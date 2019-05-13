#coding:UTF-8

from __future__ import division
from math import exp
import numpy as np
from numpy import *
from random import normalvariate#正态分布
from datetime import datetime

trainData = 'diabetes_train.txt'
testData = 'diabetes_test.txt'
featureNum = 8
max_list = []
min_list = []

def normalize(x_list,max_list,min_list):
    index = 0
    scalar_list = []
    for x in x_list:
        x_max = max_list[index]
        x_min = min_list[index]
        if x_max == x_min:
            x = 1.0    
        else:
            x = round((x-x_min)/(x_max-x_min),4)
        scalar_list.append(x)
        index += 1
    return scalar_list

def loadTrainDataSet(data):
    global max_list
    global min_list
    dataMat = []
    labelMat = []
    
    with open(data) as fr:
        for line in fr:
            currLine = line.strip().split(',')
            #lineArr = [1.0]
            lineArr = []

            for i in xrange(featureNum):
                lineArr.append(float(currLine[i]))

            dataMat.append(lineArr)

            labelMat.append(float(currLine[-1]) * 2 - 1)
    
    data_array = np.array(dataMat)
    max_list = np.max(data_array,axis=0)
    min_list = np.min(data_array,axis=0)

    scalar_dataMat = []
    for row in dataMat:
        scalar_row = normalize(row,max_list,min_list)
        scalar_dataMat.append(scalar_row)
    return scalar_dataMat, labelMat

def loadTestDataSet(data):
    global max_list
    global min_list
    dataMat = []
    labelMat = []
    
    with open(data) as fr:
        for line in fr:
            currLine = line.strip().split(',')
            lineArr = []

            for i in xrange(featureNum):
                lineArr.append(float(currLine[i]))

            dataMat.append(lineArr)

            labelMat.append(float(currLine[-1]) * 2 - 1)
    
    data_array = np.array(dataMat)

    scalar_dataMat = []
    for row in dataMat:
        scalar_row = normalize(row,max_list,min_list)
        scalar_dataMat.append(scalar_row)
    return scalar_dataMat, labelMat

def sigmoid(inx):
    return 1. / (1. + exp(-max(min(inx, 15.), -15.)))
    #return 1.0 / (1 + exp(-inx))

def stocGradAscent(dataMatrix, classLabels, k, iter):
    #dataMatrix用的是mat, classLabels是列表
    m, n = shape(dataMatrix)
    alpha = 0.01
    #初始化参数
    #w = random.randn(n, 1)#其中n是特征的个数
    w = zeros((n, 1))
    w_0 = 0.
    v = normalvariate(0, 0.2) * ones((n, k))
    
    for it in xrange(iter):
        print it
        for x in xrange(m):#随机优化，对每一个样本而言的
            inter_1 = dataMatrix[x] * v
            inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)#multiply对应元素相乘
            #完成交叉项
            interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
            
            p = w_0 + dataMatrix[x] * w + interaction#计算预测的输出
            #print "y: ",p 
            loss = sigmoid(classLabels[x] * p[0, 0]) - 1
            #print "loss: ",loss
        
            w_0 = w_0 - alpha * loss * classLabels[x]
            
            for i in xrange(n):
                if dataMatrix[x, i] != 0:
                    w[i, 0] = w[i, 0] - alpha * loss * classLabels[x] * dataMatrix[x, i]
                    for j in xrange(k):
                        v[i, j] = v[i, j] - alpha * loss * classLabels[x] * (dataMatrix[x, i] * inter_1[0, j] - v[i, j] * dataMatrix[x, i] * dataMatrix[x, i])
        
    
    return w_0, w, v

def getAccuracy(dataMatrix, classLabels, w_0, w, v):
    m, n = shape(dataMatrix)
    allItem = 0
    error = 0
    result = []
    for x in xrange(m):
        allItem += 1
        inter_1 = dataMatrix[x] * v
        inter_2 = multiply(dataMatrix[x], dataMatrix[x]) * multiply(v, v)#multiply对应元素相乘
        #完成交叉项
        interaction = sum(multiply(inter_1, inter_1) - inter_2) / 2.
        p = w_0 + dataMatrix[x] * w + interaction#计算预测的输出
        
        pre = sigmoid(p[0, 0])
        
        result.append(pre)
        
        if pre < 0.5 and classLabels[x] == 1.0:
            error += 1
        elif pre >= 0.5 and classLabels[x] == -1.0:
            error += 1
        else:
            continue
        
    
    print result
    
    return float(error) / allItem
        
   
if __name__ == '__main__':
    dataTrain, labelTrain = loadTrainDataSet(trainData)
    dataTest, labelTest = loadTestDataSet(testData)
    date_startTrain = datetime.now()
    print "开始训练"
    w_0, w, v = stocGradAscent(mat(dataTrain), labelTrain, 20, 200)
    print "训练准确性为：%f" % (1 - getAccuracy(mat(dataTrain), labelTrain, w_0, w, v))
    date_endTrain = datetime.now()
    print "训练时间为：%s" % (date_endTrain - date_startTrain)
    print "开始测试"
    print "测试准确性为：%f" % (1 - getAccuracy(mat(dataTest), labelTest, w_0, w, v))  
