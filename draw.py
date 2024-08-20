import matplotlib.pyplot as plt
import pandas as pd
import csv

mae = []
rmse = []
with open('Result/mae.csv', mode='r') as file1:
    csv_reader = csv.reader(file1)
    next(csv_reader)
    for row in csv_reader:
        mae.append(float(row[0]))

with open('Result/rmse.csv', mode='r') as file2:
    csv_reader = csv.reader(file2)
    next(csv_reader)
    for row in csv_reader:
        rmse.append(float(row[0]))

plt.rcParams["font.family"] = "Times New Roman"
x = ['Decision Tree','Random Forest','SVM','Linear Regression','XGBoost']

plt.bar(x, mae, width=0.4, alpha=0.6, label='MAE',color = 'blue')
plt.ylabel('mae')
for i in range(len(mae)):
    plt.text(i, mae[i] + 1, f'{mae[i]:.2f}', ha='center')
plt.savefig('Img/mae.png', dpi=300)


# plt.bar(x, rmse, width=0.4, alpha=0.6, label='RMSE',color = 'orange')
# plt.ylabel('rmse')
# for i in range(len(rmse)):
#     plt.text(i, rmse[i] + 1, f'{rmse[i]:.2f}', ha='center')
# plt.savefig('Img/rmse.png', dpi=300)


# plt.show()
