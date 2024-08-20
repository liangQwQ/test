import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

data = pd.read_csv('../Dataset/data.csv')
train, test = train_test_split(data, test_size=0.2, random_state=0)

result = test[['id','y']]
result.loc[:, 'y'] = 0
# 保存训练集和测试集到CSV文件
train.to_csv('../Dataset/train.csv', index=False)
test.to_csv('../Dataset/test.csv', index=False)
result.to_csv('../Dataset/result.csv', index=False)

'''
# 设置随机种子，以确保可重复性
np.random.seed(42)
# 计算数据集大小
dataset_size = len(data)
# 计算测试集大小
test_size = int(dataset_size * 0.2)
# 生成随机索引
indices = np.random.permutation(dataset_size)
# 分割索引为训练集和测试集
train_indices = indices[:-test_size]
test_indices = indices[-test_size:]
# 根据索引获取训练集和测试集
train = data.loc[train_indices]
test = data.loc[test_indices]
# 保存训练集和测试集到CSV文件
train.to_csv('train_set.csv', index=False)
test.to_csv('test_set.csv', index=False)
'''
