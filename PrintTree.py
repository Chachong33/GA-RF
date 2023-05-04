import pandas as pd 

import graphviz
data = pd.read_csv("data/data.csv")
print(data.head())
count = data.VCI.value_counts()
print(count)
#print(data.describe())
feature = data[['genger','age','education','living','drinking','smoking','tea','sleep','exercise',	
'social contect','hobby','diet','hypertension','diabetes','hyperlipemia','cardiovascular']]
target  = data[['VCI']] 
from sklearn import tree
model = tree.DecisionTreeClassifier()
model = model.fit(feature,target)


dot_data = tree.export_graphviz(model,out_file=None)
graph = graphviz.Source(dot_data)
graph.render("Tree")