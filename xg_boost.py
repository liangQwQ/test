import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from xgboost import XGBRegressor
from xgboost import plot_importance
from sklearn.feature_selection import f_regression

train = pd.read_csv('Dataset/train.csv')
test = pd.read_csv('Dataset/test.csv')
result = pd.read_csv('Dataset/result.csv')

# 获得球员年龄
today = pd.Timestamp(year=2018, month=4, day=15)
train['birth_date'] = pd.to_datetime(train['birth_date'])
train['age'] = (today - train['birth_date']).apply(lambda x: x.days) / 365.
test['birth_date'] = pd.to_datetime(test['birth_date'])
test['age'] = (today - test['birth_date']).apply(lambda x: x.days) / 365.
train = train.drop('birth_date', axis=1)
test = test.drop('birth_date', axis=1)

# 获得球员最擅长位置上的评分
positions = ['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk']
train['best_pos'] = train[positions].max(axis=1)
test['best_pos'] = test[positions].max(axis=1)

train['is_gk'] = train['gk'] > 0
test['is_gk'] = test['gk'] > 0

train = train.drop(['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk'], axis=1)
test = test.drop(['rw', 'rb', 'st', 'lw', 'cf', 'cam', 'cm', 'cdm', 'cb', 'lb', 'gk'], axis=1)

# 计算球员的身体质量指数(BMI)
train['BMI'] = 10000. * train['weight_kg'] / (train['height_cm'] ** 2)
test['BMI'] = 10000. * test['weight_kg'] / (test['height_cm'] ** 2)
train = train.drop(['weight_kg', 'height_cm'], axis=1)
test = test.drop(['weight_kg', 'height_cm'], axis=1)

# 筛选特征
list1 = ['pac', 'sho', 'pas', 'dri', 'def', 'phy']
# list1 = [item for item in train.columns.values if item not in ['y']]
X = train.loc[:, list1].values
Y = train.loc[:, 'y'].values
F_score, p_value = f_regression(X, Y)
selected_features1 = [name for name, score in zip(list1, F_score) if score > 600]
print(selected_features1)

list2 = ['skill_moves', 'weak_foot', 'work_rate_att', 'work_rate_def', 'preferred_foot', 'crossing', 'finishing',
         'heading_accuracy', 'short_passing', 'volleys', 'dribbling', 'curve', 'free_kick_accuracy', 'long_passing',
         'ball_control']
X2 = train.loc[:, list2].values
Y = train.loc[:, 'y'].values
F_score, p_value = f_regression(X2, Y)
selected_features2 = [name for name, score in zip(list2, F_score) if score > 600]
print(selected_features2)

list3 = ['acceleration', 'sprint_speed', 'agility', 'reactions', 'balance', 'shot_power', 'jumping', 'stamina',
         'strength','long_shots', 'aggression', 'interceptions', 'positioning', 'vision', 'penalties', 'marking',
         'standing_tackle', 'sliding_tackle', 'gk_diving','gk_handling', 'gk_kicking', 'gk_positioning', 'gk_reflexes']
X3 = train.loc[:, list3].values
Y = train.loc[:, 'y'].values
F_score, p_value = f_regression(X3, Y)
selected_features3 = [name for name, score in zip(list3, F_score) if score > 600]
print(selected_features3)


cols = ['club', 'league', 'potential','age', 'best_pos', 'BMI', 'is_gk']
# cols = selected_features
cols += selected_features1 + selected_features2 + selected_features3
xgb_reg = XGBRegressor(n_estimators=100, learning_rate=0.1, random_state=0)
xgb_reg.fit(train[cols], train['y'])
plot_importance(xgb_reg)
plt.show()
result['y'] = xgb_reg.predict(test[cols])

result.to_csv('Result/xgboost.csv', index=False)

true_value = test.loc[:, 'y']
pred_value = result.loc[:, 'y']

mae = abs(true_value - pred_value).mean()

print(f'Mean Absolute Error (MAE) : {mae}')
