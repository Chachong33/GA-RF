import xgboost as xgb
import pandas as pd
import numpy as np
import  random

import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from  sklearn.model_selection import  train_test_split


df = pd.read_csv('data/end_data.csv')
#随机划分训练集和测试集
X_train,X_test,y_train,y_test=train_test_split(df.loc[:,df.columns !='Target'],df['Target'],
                                               stratify=df['Target'],random_state=50)

from sklearn.metrics import accuracy_score,classification_report
from sklearn import metrics
from sklearn.metrics import roc_auc_score

#决策树ROC曲线

from sklearn import tree
tr = tree.DecisionTreeClassifier(criterion="gini",max_depth=4,random_state=0,splitter="best")
tr.fit(X_train,y_train)
#names = ['Decision Tree', # 这个就是模型标签，我们使用三个，所以需要三个标签
#        ]

#随机森林ROC曲线图
rf = RandomForestClassifier(n_estimators = 100, max_depth = 4, max_features = 3, bootstrap = True).fit(X_train, y_train)
names = ['Decision Tree',
         'Random Forest', # 这个就是模型标签，我们使用三个，所以需要三个标签

        ]


'''
xgc = xgb.XGBClassifier(random_state = 1,learning_rate = 0.2154,max_depth = 5,n_estimators = 3 )
xgc.fit(X_train,y_train)\
names = ['XGBoost', # 这个就是模型标签，我们使用三个，所以需要三个标签
        ]
'''
sampling_methods = [tr, # 这个就是训练的模型。
                   rf,
                  #  knn
                   ]

colors = ['crimson',  # 这个是曲线的颜色，几个模型就需要几个颜色哦！
          'orange',
         # 'lawngreen'
         ]



def makeROC(names, sampling_methods, colors, X_test, y_test,  dpin=100):

    plt.figure(figsize=(20, 20), dpi=dpin)

    for (name, method, colorname) in zip(names, sampling_methods, colors):
        y_test_preds = method.predict(X_test)
        y_test_predprob = method.predict_proba(X_test)[:, 1]
        fpr, tpr, thresholds = metrics.roc_curve(y_test, y_test_predprob, pos_label=1)

        plt.plot(fpr, tpr, lw=5, label='{} (AUC={:.3f})'.format(name, metrics.auc(fpr, tpr)), color=colorname)
        plt.plot([0, 1], [0, 1], '--', lw=5, color='grey')
        plt.axis('square')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.xlabel('False Positive Rate', fontsize=20)
        plt.ylabel('True Positive Rate', fontsize=20)
        plt.title('ROC Curve', fontsize=25)
        plt.legend(loc='lower right', fontsize=20)
    plt.show()
    return plt


#ROC curves
test_roc_graph = makeROC(names, sampling_methods, colors, X_test, y_test)  # 这里可以改成训练集
test_roc_graph.savefig('ROC_Train_all.png')
