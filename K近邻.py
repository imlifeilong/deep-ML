import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

columns = ['特征1', '特征2', '结论', '得分']
data = [
    [7, 8, '好', 7.3],
    [8, 6, '好', 6.7],
    [4, 8, '坏', 5.3],
    [5, 5, '坏', 5.3],
    [2, 4, '坏', 3.7],
    [8, 5, '好', 7],
    [9, 8, '好', 6.7],
]

df = pd.DataFrame(data=data, columns=columns)
print(df)
sns.set(context='talk')
# 加载字体
plt.rcParams['font.sans-serif'] = ['SimHei']
# 加载字体
sns.relplot(
    x='特征1',
    y='特征2',
    hue='结论',
    kind='scatter',
    data=df
)

for x, y, z in zip(df['特征1'], df['特征2'], df['得分']):
    plt.text(x=x, y=y, s=z)

# x y 轴从0到10
plt.xlim((0, 10))
plt.ylim((0, 10))
plt.show()

# 距离该点最近得3个点 判断该点类别
models = KNeighborsClassifier(
    n_neighbors=3
)
models.fit(
    X=df[['特征1', '特征2']],
    y=df['结论']
)
res = models.predict(X=[[4, 6], [8, 8]])
print(res)

# K近邻回归 离该点最近得三个点的得分值得平均值（K-means）
models_reg = KNeighborsRegressor(n_neighbors=3)
models_reg.fit(
    X=df[['特征1', '特征2']],
    y=df['得分']
)
reg_res = models_reg.predict(
    X=[[4, 6], [8, 8]]
)
print(reg_res)
