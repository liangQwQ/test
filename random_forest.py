# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
from datetime import date
from sklearn.ensemble import RandomForestRegressor

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


# 计算球员的身体质量指数(BMI)
train['BMI'] = 10000. * train['weight_kg'] / (train['height_cm'] ** 2)
test['BMI'] = 10000. * test['weight_kg'] / (test['height_cm'] ** 2)


# 判断一个球员是否是守门员
train['is_gk'] = train['gk'] > 0
test['is_gk'] = test['gk'] > 0


# 用多个变量准备训练随机森林
test['pred'] = 0
cols = ['height_cm', 'weight_kg', 'potential', 'BMI', 'pac',
        'phy', 'international_reputation', 'age', 'best_pos']


# 用非守门员数据训练随机森林
reg_ngk = RandomForestRegressor(random_state=100)
reg_ngk.fit(train[train['is_gk'] == False][cols], train[train['is_gk'] == False]['y'])

preds = reg_ngk.predict(test[test['is_gk'] == False][cols])
test.loc[test['is_gk'] == False, 'pred'] = preds


# 用守门员数据训练随机森林
reg_gk = RandomForestRegressor(random_state=100)
reg_gk.fit(train[train['is_gk'] == True][cols], train[train['is_gk'] == True]['y'])

preds = reg_gk.predict(test[test['is_gk'] == True][cols])
test.loc[test['is_gk'] == True, 'pred'] = preds


# 输出预测值
result['y'] = np.array(test['pred'])
result.to_csv('Result/random_forest.csv', index=False)