import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('data/end1_data.csv')  # 想要读取的文件数据
plt.figure(figsize=(11,11))
#相关系数矩阵
sns.set(font_scale=0.9)
sns.heatmap(df.corr(),cmap="Blues",annot=True)
plt.show()
#绘制矩阵关系图 ：对角线坐标为同一变量。非对角线横纵坐标变量不同，可以看它们之间的相关性
df = pd.read_csv('data/end_data.csv')  # 想要读取的文件数据
sns.set(font_scale=0.7)
sns.pairplot(df,kind='scatter',diag_kind='kde',hue='Target') #就是联合分布图构成的矩阵，
# 联合分布图不仅能看到数据属性之间的相关性，也能看到对于某个类别的某个属性值是如何分布的。
plt.show()

