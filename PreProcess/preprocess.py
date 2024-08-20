import pandas as pd
from datetime import date

data = pd.read_csv('../Dataset/original_data.csv')

# 查看概要信息
print(data.info())

# 检查缺失值
missing_values = data.isnull().sum()
print(missing_values)
# missing_values.to_csv('missing_values.csv')

# 处理缺失值
data.fillna(0, inplace=True)

# 检查重复值
# num_duplicate_rows = data.duplicated().sum()
# print(num_duplicate_rows)

# # 删除重复值
# data.drop_duplicates(subset=None,keep='first',inplace=True)
# print(data)

# 处理非数值型数据
label_mapping = {
    'Low': 0,
    'Medium': 1,
    'High': 2
}
data['work_rate_att'] = data['work_rate_att'].map(label_mapping)
data['work_rate_def'] = data['work_rate_def'].map(label_mapping)


# # 将出生年月日转化为年龄
# # today = date(2018, 4, 15)
# today = pd.Timestamp(year=2018, month=4, day=15)
# data['birth_date'] = pd.to_datetime(data['birth_date'])
# data['age'] = (today - data['birth_date']).apply(lambda x: x.days) / 365.


# # 查看数据最大值，最小值，均值
# print(data.describe())

data.to_csv('../Dataset/data.csv', index=False)