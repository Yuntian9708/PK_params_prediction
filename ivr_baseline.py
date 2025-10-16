import os
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_graphviz
# -*- coding: utf-8 -*-
import matplotlib.pyplot as plt
# import cupy
import torch
from sklearn import tree, metrics
from sklearn.model_selection import KFold, StratifiedKFold, GroupKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from sklearn.model_selection import train_test_split
from sklearn.tree import export_graphviz
from sklearn.metrics import roc_curve, auc, mean_squared_error, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor

seed = 42
np.random.seed(seed)
pk_path = './data/FVIII_hl_cl_24.xlsx'
save_path = './'
# extrapolation_path = './data/FVIII_extrapolation_data.xlsx'
df_pk = pd.read_excel(pk_path)
# df_extrapolation = pd.read_excel(extrapolation_path)
features = df_pk.iloc[:, 1:13].values
ivr_label = df_pk.iloc[:, -1].values
# hl_label = df_pk.iloc[:, -2].values
# extrapolation = df_extrapolation.iloc[:, 1:17].values
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
qt_scaler = QuantileTransformer()
x_train = features[:80, :]  # 80, 78
ivr_train = ivr_label[:80]
x_test = features[80:, :]
ivr_test = ivr_label[80:]
x_train_norm = minmax_scaler.fit_transform(x_train)
x_test_norm = minmax_scaler.transform(x_test)
cl_train_norm = std_scaler.fit_transform(ivr_train.reshape(-1, 1))
cl_test_norm = std_scaler.transform(ivr_test.reshape(-1, 1))

regressor_rf_ivr = RandomForestRegressor(n_estimators=300, max_depth=10, random_state=seed, min_samples_leaf=3)
# cl_mae = -1 * cross_val_score(regressor_rf_cl, features, cl_label, cv=5, scoring='neg_mean_absolute_error')
# cl_mse = -1 * cross_val_score(regressor_rf_cl, features, cl_label, cv=5, scoring='neg_mean_squared_error')
# print("MAE score:\n", np.mean(cl_mae))
# print("MSE score:\n", np.mean(cl_mse))
regressor_rf_ivr.fit(x_train, ivr_train)
rf_cl_pred = regressor_rf_ivr.predict(x_test)


regressor_mlp_cl = MLPRegressor(hidden_layer_sizes=(128, ), activation='relu', solver='lbfgs', learning_rate='adaptive',
                                learning_rate_init=1e-3, shuffle=True, random_state=seed, max_iter=500, batch_size=16)
regressor_mlp_cl.fit(x_train_norm, cl_train_norm)
mlp_cl_pred = regressor_mlp_cl.predict(x_test_norm)
mlp_cl_pred = std_scaler.inverse_transform(mlp_cl_pred.reshape(-1, 1))

rf_mse = mean_squared_error(ivr_test, rf_cl_pred)
rf_mae = mean_absolute_error(ivr_test, rf_cl_pred)

mlp_mse = mean_squared_error(ivr_test, mlp_cl_pred)
mlp_mae = mean_absolute_error(ivr_test, mlp_cl_pred)

rf_result = [ivr_test, rf_cl_pred]
result_column = ['真实值', '预测值']
rf_result_df = pd.DataFrame(rf_result).T
rf_result_df.columns = result_column
rf_result_df.reset_index(drop=True)
print('-'*5, 'RF预测结果', '-'*5)
print(rf_result_df)
print('MSE：', round(rf_mse, 3))
print('MAE：', round(rf_mae, 3))
# print('------Extrapolation-56h------')
# print(extrapolation_pred)


mlp_pk_pred = mlp_cl_pred.squeeze(1)
mlp_result = [ivr_test, mlp_pk_pred]
mlp_result_df = pd.DataFrame(mlp_result).T
mlp_result_df.columns = result_column
mlp_result_df = mlp_result_df.reset_index(drop=True)
print('-'*5, 'MLP预测结果', '-'*5)
print(mlp_result_df.reset_index(drop=True))
print('MSE：', round(mlp_mse, 3))
print('MAE：', round(mlp_mae, 3))

feature_names = ['Blood Group', 'Height', 'Weight', 'BMI', 'FFM', 'Age','Dose/IU', 'IU/KG','VWF-Ag',
                 'pre', 'timst', '24fviii']
importance = regressor_rf_ivr.feature_importances_
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), importance), feature_names), reverse=True))
feature_importances_df = pd.DataFrame({'feature': feature_names, 'importance': importance})
# 对特征重要性得分进行排序
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)

# 颜色映射
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_names)))

# 可视化特征重要性
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importances_df['feature'], feature_importances_df['importance'], color=colors)
ax.invert_yaxis()  # 翻转y轴，使得最大的特征在最上面
ax.set_xlabel('feature_importance', fontsize=12)  # 图形的x标签
ax.set_title('feature importance by RF regressor for IVR ', fontsize=16)
for i, v in enumerate(feature_importances_df['importance']):
    ax.text(v + 0.01, i, str(round(v, 3)), va='center', fontname='Times New Roman', fontsize=10)

# # 设置图形样式
# plt.style.use('default')
ax.spines['top'].set_visible(False)  # 去掉上边框
ax.spines['right'].set_visible(False)  # 去掉右边框
# ax.spines['left'].set_linewidth(0.5)#左边框粗细
# ax.spines['bottom'].set_linewidth(0.5)#下边框粗细
# ax.tick_params(width=0.5)
# ax.set_facecolor('white')#背景色为白色
# ax.grid(False)#关闭内部网格线

# 保存图形
# plt.savefig('./特征重要性.jpg', dpi=400, bbox_inches='tight')
plt.show()
