import pandas as pd 
import matplotlib.pylab as plt
#import xlrd
#导入决策树函数库
from sklearn.tree import DecisionTreeClassifier
#导入随机森林函数库
from sklearn.ensemble import RandomForestClassifier
#导入划分训练集和测试集的函数库、交叉验证函数库
from sklearn.model_selection import train_test_split,cross_val_predict,cross_val_score
#导入精度评分、混淆矩阵函数库
from sklearn.metrics import accuracy_score,confusion_matrix

name = pd.read_csv(open('data/data.csv','r'))
Xtrain,Xtest,Ytrain,Ytest = train_test_split(name.values[:,:-1],name['VCI'],test_size=0.3,random_state=3)
print(Xtrain.shape,Xtest.shape)
print(Ytrain.shape,Ytest.shape)
# print(news.keys())
# test = pd.DataFrame()

# print(news['Sepal.Width'])
tree_0 = DecisionTreeClassifier(random_state=3)
#random_state设置随机种子，保证每一次都是同一个随机数。若为0或不填，则每次得到数据都不一样
#构建决策树
tree_0.fit(Xtrain,Ytrain)
#输出决策树的测试精度
print(tree_0.score(Xtest,Ytest))


rfc_0=RandomForestClassifier(random_state=3,n_estimators=10)
#n_estimators:树的个数，太小容易欠拟合，太大容易过拟合
rfc_0.fit(Xtrain,Ytrain)
print(rfc_0.score(Xtest,Ytest))
#用训练的模型执行预测
pred_rf = rfc_0.predict(Xtest)
#打印出预测错误的结果看看
for i in range(len(Ytest)):
   if Ytest.iloc[i]!=pred_rf[i]:
        print(i,"Actual class is",Ytest.iloc[i],"and predicted is",pred_rf[i])

#输出预测精度
print(rfc_0.score(Xtest,Ytest))
#打印混淆矩阵
print('confusion_matrix',confusion_matrix(Ytest,pred_rf))

#k-fold交叉验证，并查看平均精度得分
X=name.values[:,:-1]
Y=name['VCI']
scores = cross_val_predict(rfc_0,X,Y,cv=10)
print("The accurancy scores for the iterations are {}".format(scores))
print(accuracy_score(scores, Y))


#随迭代次数变化的交叉验证精度的变化情况
#设置迭代次数的范围
k_range = range(1,40)
#设置变量cv_scores,存储每个模型的精度值
cv_scores = []
#计算不同迭代次数时的精度值，存取cv_scores中
for n in k_range:
    rfc = RandomForestClassifier(n_estimators=n,n_jobs=-1)
    scores = cross_val_score(rfc,X,Y,cv=10,scoring='accuracy')
    cv_scores.append(scores.mean())
#绘制精度随k值变化的曲线图
plt.plot(k_range,cv_scores)
plt.xlabel('K')
plt.ylabel('Accurancy')
plt.show()

rfc_f = RandomForestClassifier(n_estimators=24,n_jobs=-1)
scoresf = cross_val_score(rfc_f,X,Y,cv=10,scoring='accuracy')
print(scoresf.mean())