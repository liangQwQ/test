import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

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


# 用‘潜力’，‘国际知名度’，‘年龄’，‘最擅长位置评分’这四个变量来建立SVR模型
cols = ['potential', 'international_reputation', 'age', 'best_pos']

clf = make_pipeline(StandardScaler(), LinearRegression())
clf.fit(train[cols], train['y'])

lower_bound = 0
predictions = clf.predict(test[cols])
result['y'] = np.clip(predictions, lower_bound, None)

# result['y'] = clf.predict(test[cols])

result.to_csv('Result/lr.csv', index=False)

true_value = test.loc[:,'y']
pred_value = result.loc[:,'y']

mae = abs(true_value - pred_value).mean()

print(f'Mean Absolute Error (MAE) : {mae}')
