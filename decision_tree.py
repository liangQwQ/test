# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import date
from sklearn.tree import DecisionTreeRegressor


# 读取数据
train = pd.read_csv('Dataset/train.csv')
test = pd.read_csv('Dataset/test.csv')
result = pd.read_csv('Dataset/result.csv')

# 获得球员年龄
today = pd.Timestamp(year=2018, month=4, day=15)
train['birth_date'] = pd.to_datetime(train['birth_date'])
train['age'] = (today - train['birth_date']).apply(lambda x: x.days) / 365.
test['birth_date'] = pd.to_datetime(test['birth_date'])
test['age'] = (today - test['birth_date']).apply(lambda x: x.days) / 365.

# 获得球员最擅长位置上的评分
positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']

train['best_pos'] = train[positions].max(axis=1)
test['best_pos'] = test[positions].max(axis=1)


# 用‘潜力’，‘国际知名度’，‘年龄’，‘最擅长位置评分’这四个变量来建立决策树模型
cols = ['potential', 'international_reputation', 'age', 'best_pos']

reg = DecisionTreeRegressor(random_state=100)
reg.fit(train[cols], train['y'])


# 输出预测值
result['y'] = reg.predict(test[cols])
result.to_csv('Result/decision_tree.csv', index=False)