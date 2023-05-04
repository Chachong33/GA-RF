import xgboost as xgb
import pandas as pd
import numpy as np
import  random

import matplotlib.pyplot as plt
from  sklearn.model_selection import  train_test_split


df = pd.read_csv('data/end_data.csv')
#随机划分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(df.loc[:,df.columns !='Target'],df['Target'],
                                               stratify=df['Target'],random_state=50)

name = 'RandomForest'

#绘制混淆矩阵
from sklearn.metrics import  confusion_matrix
from sklearn.metrics import  ConfusionMatrixDisplay
def makeConfushinMatrix(model):

    y_pre=model.predict(X_train)
    confusion=confusion_matrix(y_train,y_pre)
    plt.imshow(confusion,cmap=plt.cm.YlOrRd)
    indices = range(len(confusion))
    plt.xticks(indices,['No','Yes'])
    plt.colorbar()
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title(name + " Train Accuracy")
    for first_index in range (len(confusion)):
        for second_index in range(len(confusion)):
            plt.text(first_index,second_index,confusion[second_index][first_index],va='center',ha='center')
    plt.show()
    y_tpre=model.predict(X_test)
    confusion=confusion_matrix(y_test,y_tpre)
    print(confusion)
    plt.imshow(confusion,cmap=plt.cm.YlOrRd)
    indices = range(len(confusion))
    plt.xticks(indices,['No','Yes'])
    plt.colorbar()
    plt.xlabel("True Labels")
    plt.ylabel("Predicted Labels")
    plt.title(name+' Test Accuracy')
    for first_index in range (len(confusion)):
        for second_index in range(len(confusion)):
            plt.text(first_index,second_index,confusion[second_index][first_index],va='center',ha='center'
                     )
    plt.show()



xgc = xgb.XGBClassifier(random_state = 1,learning_rate = 0.2154,max_depth = 5,n_estimators = 3 )
xgc.fit(X_train,y_train)
print(xgc)
makeConfushinMatrix(xgc)
