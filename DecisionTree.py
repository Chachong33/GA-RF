# -*- coding:utf-8 -*- 

from numpy import *
import numpy as np
import pandas as pd
from math import log
import operator
import re
 
# 计算数据集的熵
def calcShannonEnt(dataSet):
    numEntries = len(dataSet)
    labelCounts = {}
    # 给所有可能分类创建字典
    for featVec in dataSet:
        currentLabel = featVec[-1]
        if currentLabel not in labelCounts.keys():
            labelCounts[currentLabel] = 0
        labelCounts[currentLabel] += 1
    shannonEnt = 0.0
    # 以2为底数计算香农熵
    for key in labelCounts:
        prob = float(labelCounts[key]) / numEntries
        shannonEnt -= prob * log(prob, 2)
    return shannonEnt

# 对离散变量划分数据集，取出该特征取值为value的所有样本
def splitDataSet(dataSet, axis, value):
    retDataSet = []
    for featVec in dataSet:
        if featVec[axis] == value:
            reducedFeatVec = featVec[:axis]
            reducedFeatVec.extend(featVec[axis + 1:])
            retDataSet.append(reducedFeatVec)
    return retDataSet


# 对连续变量划分数据集，direction规定划分的方向，
# 决定是划分出小于value的数据样本还是大于value的数据样本集
def splitContinuousDataSet(dataSet, axis, value, direction):
    retDataSet = []
    for featVec in dataSet:
        if direction == 0:
            if featVec[axis] > value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
        else:
            if featVec[axis] <= value:
                reducedFeatVec = featVec[:axis]
                reducedFeatVec.extend(featVec[axis + 1:])
                retDataSet.append(reducedFeatVec)
    return retDataSet


# 选择最好的数据集划分方式
def chooseBestFeatureToSplit(dataSet, labels):
    numFeatures = len(dataSet[0]) - 1
    # print(numFeatures)
    baseEntropy = calcShannonEnt(dataSet)# 计算数据集的香农熵
    # print(baseEntropy)
    bestInfoGain = 0.0
    bestFeature = -1
    bestSplitDict = {}
    for i in range(numFeatures):
        featList = [example[i] for example in dataSet]#每次记录一列的值
        # 对连续型特征进行处理
        if type(featList[0]).__name__ == 'float' or type(featList[0]).__name__ == 'int':
            # 产生n-1个候选划分点
            sortfeatList = sorted(featList)
            splitList = []
            for j in range(len(sortfeatList) - 1):
                splitList.append((sortfeatList[j] + sortfeatList[j + 1]) / 2.0)

            bestSplitEntropy = 10000
            slen = len(splitList)
            # 求用第j个候选划分点划分时，得到的信息熵，并记录最佳划分点
            for j in range(slen):
                value = splitList[j]
                newEntropy = 0.0
                subDataSet0 = splitContinuousDataSet(dataSet, i, value, 0)
                subDataSet1 = splitContinuousDataSet(dataSet, i, value, 1)
                prob0 = len(subDataSet0) / float(len(dataSet))
                newEntropy += prob0 * calcShannonEnt(subDataSet0)
                prob1 = len(subDataSet1) / float(len(dataSet))
                newEntropy += prob1 * calcShannonEnt(subDataSet1)
                if newEntropy < bestSplitEntropy:
                    bestSplitEntropy = newEntropy
                    bestSplit = j
            # 用字典记录当前特征的最佳划分点
            bestSplitDict[labels[i]] = splitList[bestSplit]
            infoGain = baseEntropy - bestSplitEntropy
        # 对离散型特征进行处理
        else:
            uniqueVals = set(featList)
            newEntropy = 0.0
            # 计算该特征下每种划分的信息熵
            for value in uniqueVals:
                subDataSet = splitDataSet(dataSet, i, value)
                prob = len(subDataSet) / float(len(dataSet))
                newEntropy += prob * calcShannonEnt(subDataSet)
            infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeature = i
    # 若当前节点的最佳划分特征为连续特征，则将其以之前记录的划分点为界进行二值化处理
    # 即是否小于等于bestSplitValue
    # print(labels[bestFeature],bestInfoGain)
    if type(dataSet[0][bestFeature]).__name__ == 'float' or type(dataSet[0][bestFeature]).__name__ == 'int':
        # print(bestFeature)
        bestSplitValue = bestSplitDict[labels[bestFeature]]
        labels[bestFeature] = labels[bestFeature] + '<=' + str(bestSplitValue)
        for i in range(shape(dataSet)[0]):
            if dataSet[i][bestFeature] <= bestSplitValue:
                dataSet[i][bestFeature] = 1
            else:
                dataSet[i][bestFeature] = 0
    return bestFeature


# 特征若已经划分完，节点下的样本还没有统一取值，则需要进行投票
def majorityCnt(classList):
    classCount = {}
    #计算每个值的对应人数
    for vote in classList:
        if vote not in classCount.keys():
            classCount[vote] = 0
        classCount[vote] += 1
    #返回人最多的值
    return max(classCount)

# 主程序，递归产生决策树
def createTree(dataSet, labels, data_full, labels_full):

    classList = [example[-1] for example in dataSet]   #列表存储最后一个元素
    # print(classList)
    # print(classList[0],classList.count(classList[0]),len(classList))
    if classList.count(classList[0]) == len(classList) :#如果所有的值都相同，直接返回classList【0】
        return classList[0]
    if len(classList)<30:
        return majorityCnt(classList)
        # if(classList.count(1)>=classList.count(2)):
        #     return 1
        # return 2

    bestFeat = chooseBestFeatureToSplit(dataSet, labels)# 选择最好的数据集划分方式的值
    # print(bestFeat)
    bestFeatLabel = labels[bestFeat]#最好划分值的列名
    myTree = {bestFeatLabel: {}}
    featValues = [example[bestFeat] for example in dataSet]#最好划分项所有值
    uniqueVals = set(featValues)#划分项所有的值的集合

    listleng=len(featValues)
    print(bestFeatLabel)
    for value in uniqueVals:
        valuecount=featValues.count(value)
        print(value,':',valuecount,valuecount/listleng)

    # print(dataSet[0][bestFeat])
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        currentlabel = labels_full.index(labels[bestFeat])
        featValuesFull = [example[currentlabel] for example in data_full]
        uniqueValsFull = set(featValuesFull)
    # print(uniqueVals)
    del (labels[bestFeat])
    # 针对bestFeat的每个取值，划分出一个子树。
    for value in uniqueVals:
        subLabels = labels[:]

        if type(dataSet[0][bestFeat]).__name__ == 'str':
            uniqueValsFull.remove(value)
            # print(num+1)
            # print('diyige')
            # print(type(dataSet[0][bestFeat]).__name__ == 'str')
        # print(splitDataSet(dataSet, bestFeat, value))
        myTree[bestFeatLabel][value]=createTree(splitDataSet(dataSet, bestFeat, value), subLabels, data_full, labels_full)
        # print(value,featValues.count(value)/len(featValues))
        # print(featValues.count(value)/len(featValues))
    if type(dataSet[0][bestFeat]).__name__ == 'str':
        for value in uniqueValsFull:
            myTree[bestFeatLabel][value] = majorityCnt(classList)
            # print(type(dataSet[0][bestFeat]).__name__ == 'str')
            # print("dierge")
            # print(num+222)
    return myTree

import matplotlib.pyplot as plt

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round4", fc="0.8")
arrow_args = dict(arrowstyle="<-")

# 计算树的叶子节点数量
def getNumLeafs(myTree):
    numLeafs = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            numLeafs += getNumLeafs(secondDict[key])
        else:
            numLeafs += 1
    return numLeafs

# 计算树的最大深度
def getTreeDepth(myTree):
    maxDepth = 0
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    secondDict = myTree[firstStr]
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            thisDepth = 1 + getTreeDepth(secondDict[key])
        else:
            thisDepth = 1
        if thisDepth > maxDepth:
            maxDepth = thisDepth
    return maxDepth

# 画节点
def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xycoords='axes fraction', \
                            xytext=centerPt, textcoords='axes fraction', va="center", ha="center", \
                            bbox=nodeType, arrowprops=arrow_args)

# 画箭头上的文字
def plotMidText(cntrPt, parentPt, txtString):
    lens = len(txtString)
    xMid = (parentPt[0] + cntrPt[0]) / 2.0 - lens * 0.002
    yMid = (parentPt[1] + cntrPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, txtString)

def plotTree(myTree, parentPt, nodeTxt):
    numLeafs = getNumLeafs(myTree)
    depth = getTreeDepth(myTree)
    firstSides = list(myTree.keys())
    firstStr = firstSides[0]
    cntrPt = (plotTree.x0ff + (1.0 + float(numLeafs)) / 2.0 / plotTree.totalW, plotTree.y0ff)
    plotMidText(cntrPt, parentPt, nodeTxt)
    plotNode(firstStr, cntrPt, parentPt, decisionNode)
    secondDict = myTree[firstStr]
    plotTree.y0ff = plotTree.y0ff - 1.0 / plotTree.totalD
    for key in secondDict.keys():
        if type(secondDict[key]).__name__ == 'dict':
            plotTree(secondDict[key], cntrPt, str(key))
        else:
            plotTree.x0ff = plotTree.x0ff + 1.0 / plotTree.totalW
            plotNode(secondDict[key], (plotTree.x0ff, plotTree.y0ff), cntrPt, leafNode)
            plotMidText((plotTree.x0ff, plotTree.y0ff), cntrPt, str(key))
    plotTree.y0ff = plotTree.y0ff + 1.0 / plotTree.totalD

#展示决策树
def createPlot(inTree):
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    axprops = dict(xticks=[], yticks=[])
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeafs(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.x0ff = -0.5 / plotTree.totalW
    plotTree.y0ff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

#
# ##################################################
#
# # 由于在Tree中，连续值特征的名称以及改为了  feature<=value的形式
# # 因此对于这类特征，需要利用正则表达式进行分割，获得特征名以及分割阈值
# def classify(inputTree, featLabels, testVec):
#     firstStr = inputTree.keys()[0]
#     if '<=' in firstStr:
#         featvalue = float(re.compile("(<=.+)").search(firstStr).group()[2:])
#         featkey = re.compile("(.+<=)").search(firstStr).group()[:-2]
#         secondDict = inputTree[firstStr]
#         featIndex = featLabels.index(featkey)
#         if testVec[featIndex] <= featvalue:
#             judge = 1
#         else:
#             judge = 0
#         for key in secondDict.keys():
#             if judge == int(key):
#                 if type(secondDict[key]).__name__ == 'dict':
#                     classLabel = classify(secondDict[key], featLabels, testVec)
#                 else:
#                     classLabel = secondDict[key]
#     else:
#         secondDict = inputTree[firstStr]
#         featIndex = featLabels.index(firstStr)
#         for key in secondDict.keys():
#             if testVec[featIndex] == key:
#                 if type(secondDict[key]).__name__ == 'dict':
#                     classLabel = classify(secondDict[key], featLabels, testVec)
#                 else:
#                     classLabel = secondDict[key]
#     return classLabel
#
#
# # 测试决策树正确率
# def testing(myTree, data_test, labels):
#     error = 0.0
#     for i in range(len(data_test)):
#         if classify(myTree, labels, data_test[i]) != data_test[i][-1]:
#             error += 1
#             # print 'myTree %d' %error
#     return float(error)
#
#
# # 测试投票节点正确率
# def testingMajor(major, data_test):
#     error = 0.0
#     for i in range(len(data_test)):
#         if major != data_test[i][-1]:
#             error += 1
#             # print 'major %d' %error
#     return float(error)
#
#
# # 后剪枝
# def postPruningTree(inputTree, dataSet, data_test, labels):
#     f= list(inputTree.keys())
#     firstStr=f[0]
#     # print(firstStr)
#     secondDict = inputTree[firstStr]
#     classList = [example[-1] for example in dataSet]
#     featkey = copy(firstStr)
#     if '<=' in firstStr:
#         featkey = re.compile("(.+<=)").search(firstStr).group()[:-2]
#         featvalue = float(re.compile("(<=.+)").search(firstStr).group()[2:])
#     labelIndex = labels.index(featkey)
#     temp_labels = copy(labels)
#     del (labels[labelIndex])
#     for key in secondDict.keys():
#         if type(secondDict[key]).__name__ == 'dict':
#             if type(dataSet[0][labelIndex]).__name__ == 'str':
#                 inputTree[firstStr][key] = postPruningTree(secondDict[key], \
#                                                            splitDataSet(dataSet, labelIndex, key),
#                                                            splitDataSet(data_test, labelIndex, key),
#                                                            copy(labels))
#             else:
#                 inputTree[firstStr][key] = postPruningTree(secondDict[key], \
#                                                            splitContinuousDataSet(dataSet, labelIndex, featvalue, key), \
#                                                            splitContinuousDataSet(data_test, labelIndex, featvalue,
#                                                                                   key), \
#                                                            copy(labels))
#     if testing(inputTree, data_test, temp_labels) <= testingMajor(majorityCnt(classList), data_test):
#         return inputTree
#     return majorityCnt(classList)
# #################################################

df = pd.read_csv(open('data/data.csv','r'))
# print(df)
data = df.values[:, 1:].tolist()#表中元素值
# datatest=df.values[:, 1:].tolist()
# print(data)
data_full = data[:]
# print(data_full)
labels = df.columns.values[1:-1].tolist()#记录表头
# print(labels)
labels_full = labels[:]
# print(labels_full)
myTree = createTree(data, labels, data_full, labels_full)
# print(myTree)
# myTree=postPruningTree(myTree,data,datatest,labels_full)
createPlot(myTree)

print("success")