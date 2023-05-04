# 随即森林 
import pandas as pd
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
from random import seed
from random import randrange
from math import log
import numpy as np

# 分割数据集，进行交叉验证
def spilt_data(data,n_folds):
    fold_size = int(len(data)/n_folds)
    split_data = []
    # 存放分割后的结果
    cdata = data
    for i in range(n_folds):
        fold=[]
        while len(fold) < fold_size:
            tmp = randrange(len(cdata))
            fold.append(cdata.pop(tmp))
        split_data.append(fold)
    return split_data

# 按比例有放回地抽取数据，形成训练集
def subsample(train_set,ratio):
    sub=[]
    sublen = round(len(train_set)*ratio)
    # 模拟随机森林有放回的抽样
    while len(sub)<sublen:
        index=randrange(len(trian_set)-1)
        sub.append(trian_set[index])
    return sub

# 按照划分类别及其值进行划分类别
def data_spilt(train_set,index,value):
    left=[]
    right=[]
    for row in train_set:
        if row[index] < value:
            left.append(row)
        else:
            right.append(row)
    return left,right

# 基于Gini指数计算分割loss
def get_spilt_loss(left,right,labels):
    loss=0.0
    for label in labels:
        leftsize = len(left)
        # 防止除数为0
        if leftsize != 0 :
            p = [row[-1] for row in left].count(label) / float(leftsize)
            loss += (p * (1.0 - p))
        rightsize = len(right)
        if rightsize != 0 :
            p = [row[-1] for row in right].count(label) / float(rightsize)
            loss += (p * (1.0 - p))
    return loss

#随机抽取n个特征，并且对n个特征分割，计算寻找最优分割特征
def get_best_spilt(train_set,n_features):
    features=[]
    # 求出类别结果一共有几类，并且转换为列表
    labels = list(set(row[-1] for row in train_set))
    rindex,rvalue,rloss,rleft,rright = 9999,9999,9999,None,None
    # 随机抽取n个分类特征
    while len(features) < n_features:
        index = randrange(len(train_set[0])-2)
        if index not in features:
            features.append(index)
    for index in features:
        feature_count = list(set(row[index] for row in train_set))
        for i in feature_count:
            left,right = data_spilt(train_set,index,i)
            loss = get_spilt_loss(left,right,labels)
            if loss < rloss:
                rindex,rvalue,rloss,rleft,rright = index,i,loss,left,right
    return {'index':rindex,'value':rvalue,'left':rleft,'right':rright}
# 返回一个元组，包括属性，属性分割的值，左结点列表，右节点列表

#返回分类好的节点的类别（标签）
def decide_label(final):
    res = [row[-1] for row in final]
    return max(set(res),key=res.count)
# 按照res中数量最多的类作为输出

# 递归建立决策树
def sub_spilt(root,n_features,max_depth,min_size,depth):
    left = root['left']
    right = root['right']
    # 删掉原来的左右分支，构造新的
    del(root['left'])
    del(root['right'])
    # 如果左分支或者右分支为空的话，说明已经分好类了
    if not left or not right:
        root['left'] = root['right'] = decide_label(left+right)
        return
    # 如果当前深度已经大于最大深度了，则左右分支不再继续分割
    if depth>max_depth:
        root['left'] = decide_label(left)
        root['right'] = decide_label(right)
        return
    # 如果左节点列表的数目已经小于最小类别数目了，则停止分割
    if len(left) < min_size:
        root['left'] = decide_label(left)
    else:
        # 递归建立左子树
        root['left'] = get_best_spilt(left,n_features)
        sub_spilt(root['left'],n_features,max_depth,min_size,depth+1)
    if len(right) < min_size:
        root['right'] = decide_label(right)
    else:
        root['right'] = get_best_spilt(right,n_features)
        sub_spilt(root['right'],n_features,max_depth,min_size,depth+1)

#构建决策树：
def build_tree(train_set,n_features,max_depth,min_size):
    # 先找到最好的分割树型及分割值
    root = get_best_spilt(train_set,n_features)
    # 按照找到的分割节点进行递归建立决策树
    sub_spilt(root,n_features,max_depth,min_size,1)
    return root

#预测测试集的结果
def predict(estimator,row):
    if row[estimator['index']] < estimator['value']:
        if isinstance(estimator['left'],dict):
            # 如果存在左子树，则递归地进行查找
            return predict(estimator['left'],row)
        else:
            return estimator['left']
    else:
        if isinstance(estimator['right'],dict):
            return predict(estimator['right'],row)
        else:
            return estimator['right']

# n颗决策树进行投票，返回投票最多的结果
def bagging_predict(estimators,row):
    predict_labels = [predict(estimator,row) for estimator in estimators]
    return max(set(predict_labels),key = predict_labels.count)

# 构建随机森林
def random_forest(train_set,test_set,ratio,n_features,max_depth,min_size,n_estimators):
    estimators = []
    for i in range(n_estimators):
        trainset = subsample(train_set,ratio)
        estimator = build_tree(trainset,n_features,max_depth,min_size)
        estimators.append(estimator)
    predict_labels = [bagging_predict(estimators,row) for row in test_set]
    return predict_labels

# 计算准确率
def get_accuracy(predict_labels,actual_labels):
    acc = 0
    for i in range(len(actual_labels)):
        if actual_labels[i] == predict_labels[i]:
            acc+=1
    return acc/float(len(actual_labels))

if __name__ == '__main__':
    seed(10)
    rd = pd.read_csv('data/data.csv')
    rd.drop('genger',axis=1)
    td = np.array(rd)
    data = td.tolist()
    # print(td)
    # print(len(td))
    accuracy_scores = []
    max_depth = 11
    # 最大深度
    min_size = 2
    # 最小叶节点数
    ratio = 1
    # 随机选择训练集数目
    n_features = 4
    # 随机选择的划分节点数目
    n_estimators = 110
    # 树的数目
    n_folds = 10
    # 交叉验证分割数
    folds = spilt_data(data, n_folds)
    for fold in folds:
        train_set = folds[:]
        train_set.remove(fold)
        trian_set = sum(train_set,[])
        test_set=[]
        for row in fold:
            row_copy = list(row)
            row_copy[-1] = None
            #因为要验证，所以将最后一个（表示类别的）元素设为None
            test_set.append(row_copy)
        actual_labels = [row[-1] for row in fold]
        # 保存实际的类别，与预测的进行比对
        predict_labels = random_forest(train_set,test_set,ratio,n_features,max_depth,min_size,n_estimators)
        pacc = get_accuracy(predict_labels,actual_labels)
        accuracy_scores.append(pacc)
    print('Accuracy:%s' % accuracy_scores)
    print('average score:%s' % (sum(accuracy_scores)/float(len(accuracy_scores))))