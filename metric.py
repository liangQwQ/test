import pandas as pd

true_data = pd.read_csv('Dataset/test.csv')
decision_tree_data = pd.read_csv('Result/decision_tree.csv')
random_forest_data = pd.read_csv('Result/random_forest.csv')
svm_data = pd.read_csv('Result/svm.csv')
lr_data = pd.read_csv('Result/lr.csv')
xgboost_data = pd.read_csv('Result/xgboost.csv')

true_value = true_data.loc[:,'y']
decision_tree_value = decision_tree_data.loc[:,'y']
random_forest_value = random_forest_data.loc[:,'y']
svm_value = svm_data.loc[:,'y']
lr_value = lr_data.loc[:,'y']
xgboost_value = xgboost_data.loc[:,'y']

# 计算MAE
mae1 = abs(true_value - decision_tree_value).mean()
mae2 = abs(true_value - random_forest_value).mean()
mae3 = abs(true_value - svm_value).mean()
mae4 = abs(true_value - lr_value).mean()
mae5 = abs(true_value - xgboost_value).mean()

# 计算RMSE
rmse1 = ((true_value - decision_tree_value) ** 2).mean() ** 0.5
rmse2 = ((true_value - random_forest_value) ** 2).mean() ** 0.5
rmse3 = ((true_value - svm_value) ** 2).mean() ** 0.5
rmse4 = ((true_value - lr_value) ** 2).mean() ** 0.5
rmse5 = ((true_value - xgboost_value) ** 2).mean() ** 0.5


print(f'Mean Absolute Error (MAE) for decision_tree: {mae1}')
print(f'Mean Absolute Error (MAE) for random_forest: {mae2}')
print(f'Mean Absolute Error (MAE) for svm: {mae3}')
print(f'Mean Absolute Error (MAE) for lr: {mae4}')
print(f'Mean Absolute Error (MAE) for xgboost: {mae5}')

print("RMSE for Decision Tree: ", rmse1)
print("RMSE for Random Forest: ", rmse2)
print("RMSE for SVM: ", rmse3)
print("RMSE for Linear Regression: ", rmse4)
print("RMSE for XGBoost: ", rmse5)

mae = pd.DataFrame([mae1,mae2,mae3,mae4,mae5])
rmse = pd.DataFrame([rmse1,rmse2,rmse3,rmse4,rmse5])

mae.to_csv('Result/mae.csv', index=False)
rmse.to_csv('Result/rmse.csv', index=False)