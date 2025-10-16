import os

import numpy as np
import pandas as pd
from matplotlib import pyplot
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
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
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from xgboost import XGBRegressor, plot_importance, plot_tree

seed = 14202
np.random.seed(seed)
# pk_path = './data/FVIII_hl_ivr.xlsx'
# pk_path = './data/FVIII_hl_ivr_324.xlsx'
pk_path = './data/FVIII_hl_ivr_24.xlsx'
save_path = './'
# extrapolation_path = './data/FVIII_extrapolation_data.xlsx'
df_pk = pd.read_excel(pk_path)
# df_extrapolation = pd.read_excel(extrapolation_path)
# features = df_pk.iloc[:, 1:17].values  # 三个点
# features = df_pk.iloc[:, 1:15].values  # 两个点
features = df_pk.iloc[:, 1:13].values  # 一个点
hl_label = df_pk.iloc[:, -2].values
hl_label = hl_label.astype('float')
# hl_label = df_pk.iloc[:, -2].values
# extrapolation = df_extrapolation.iloc[:, 1:17].values
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
qt_scaler = QuantileTransformer()
norm_features = minmax_scaler.fit_transform(features)
norm_hl = std_scaler.fit_transform(hl_label.reshape(-1, 1))


# K-Fold split
fold = 5


def kfold_val(fold, model):
    rmse_list = []
    mae_list = []
    mape_list = []
    r2_list = []
    adjusted_r2_list = []
    prediction_list = []

    kf = KFold(n_splits=fold)
    for train_index, test_index in kf.split(features):
        # print('Test idx:', test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = hl_label[train_index], hl_label[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        n = len(hl_label)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        adjust_r2 = 1 - (1 - r2) * (n - 1) / (n - 16 - 1)  # 11为特征数

        # print('Testing results: \n', np.round(y_pred.reshape(-1, 1).transpose(0, 1), 3))
        # print('Fold MAE: {:.3f}'.format(mae))
        # print('Fold RMSE : {:.3f}'.format(rmse))
        # print('Fold MAPE: {:.3f}'.format(mape))
        # print('Fold R2: {:.3f}'.format(r2))

        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        r2_list.append(r2)
        adjusted_r2_list.append(adjust_r2)
        prediction_list.append(y_pred)

    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)
    avg_r2 = np.mean(r2_list)
    avg_adjusted_r2 = np.mean(adjusted_r2_list)

    print('-' * 5)
    print('Avg MAE : {:.3f}'.format(avg_mae))
    print('Avg RMSE : {:.3f}'.format(avg_rmse))
    # print('Avg MAPE: {:.3f}'.format(avg_mape))
    print('Avg R2 : {:.3f}'.format(avg_r2))
    # print('Avg Adjusted R2 : {}'.format(avg_adjusted_r2))
    prediction_list = np.hstack(prediction_list)
    return prediction_list, model


def kfold_val_norm(fold, model):
    rmse_list = []
    mae_list = []
    mape_list = []
    r2_list = []
    # adjusted_r2_list = []

    kf = KFold(n_splits=fold, random_state=42, shuffle=True)
    for train_index, test_index in kf.split(norm_features):
        # print('Test idx:', test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = norm_hl[train_index], norm_hl[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        # n = len(norm_ivr)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        # adjust_r2 = 1 - (1 - r2) * (n - 1) / (n - 11 - 1)  # 11为特征数
        y_pred = std_scaler.inverse_transform((y_pred).reshape(-1, 1))
        # print('Testing results: \n', np.round(y_pred.transpose(0, 1), 3))
        # print('Fold MAE: {:.3f}'.format(mae))
        # print('Fold RMSE : {:.3f}'.format(rmse))
        # print('Fold MAPE: {:.3f}'.format(mape))
        # print('Fold R2: {:.3f}'.format(r2))

        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        r2_list.append(r2)
        # adjusted_r2_list.append(adjust_r2)

    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)
    avg_r2 = np.mean(r2_list)
    # avg_adjusted_r2 = np.mean(adjusted_r2_list)

    print('-' * 5)
    print('Avg MAE : {:.3f}'.format(avg_mae))
    print('Avg RMSE : {:.3f}'.format(avg_rmse))
    # print('Avg MAPE: {:.3f}'.format(avg_mape))
    print('Avg R2 : {:.3f}'.format(avg_r2))
    # print('Avg Adjusted R2 : {}'.format(avg_adjusted_r2))


def curve_fitting(prediction_list, model_name):
    scatter_parameter = np.polyfit(prediction_list, hl_label, 1)
    corr,_ = pearsonr(prediction_list, hl_label)
    print('scatter parameter:', scatter_parameter)
    ivr_label2 = scatter_parameter[0] * prediction_list + scatter_parameter[1]
    plt.scatter(prediction_list, hl_label)
    plt.plot(prediction_list, ivr_label2, color='g')
    plt.title(model_name, fontsize=14)
    plt.xlabel('Predicted half-life Value', fontsize=14)
    plt.ylabel('Observed half-life Value', fontsize=14)
    plt.text(0.85, 0.05, f'y = {scatter_parameter[0]:.2f}x + {scatter_parameter[1]:.2f}',
             transform=plt.gca().transAxes, fontsize=14, color='black', ha='right')
    plt.text(0.02, 0.92, f'Correlation Coefficient: {corr:.2f}', transform=plt.gca().transAxes, fontsize=12,
             color='black')


# RF
rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=seed, min_samples_leaf=3)
# XGB
xgb = XGBRegressor(max_depth=20, learning_rate=0.1, n_estimators=100, random_state=seed)
# Lasso
lasso = Lasso(alpha=0.15, random_state=seed)
# Ridge
ridge = Ridge(alpha=100, random_state=seed)
# Bagging
# bagging = BaggingRegressor(n_estimators=100, max_features=9)
# MLP
mlp = MLPRegressor(hidden_layer_sizes=(64, ), activation='relu', solver='lbfgs', learning_rate='adaptive',
                   learning_rate_init=1e-3, shuffle=False, random_state=seed, max_iter=500, batch_size=16)
# kfold_val_norm(fold, mlp)

model_list = [rf, xgb, lasso, ridge, mlp]
name_list = ['Random Forest', 'XGBoost', 'Lasso Net', 'Ridge', 'MLP']

idx = 0
for model in model_list:
    print('**********', name_list[idx], '**********')
    if name_list[idx] == 'MLP':
        kfold_val_norm(fold, model)
    else:
        kfold_val(fold, model)
    idx += 1

'''
rf_prediction, rf = kfold_val(fold, rf)
xgb_prediction, _ = kfold_val(fold, xgb)
lasso_prediction, lasso = kfold_val(fold, lasso)
ridge_prediction, ridge = kfold_val(fold, ridge)
mlp_prediction = [14.052,12.959,20.117,12.554,10.372,15.072,12.394,15.306,12.947,13.913,15.132,12.165,13.806,12.693,
16.372,12.161,14.465,20.755,16.016,15.503,14.491,13.594,14.552,14.688,13.479,15.143,14.72,14.716,18.841,10.334,8.836,
13.046,13.926,13.997,13.231,11.505,15.724,21.256,12.553,15.228,15.896,16.639,22.116,10.89,13.793,14.329,12.72,10.703,
13.246,13.292,14.628,13.395,11.322,14.829,14.331,18.913,11.335,19.946,13.547,13.612,14.478,17.32,13.836,10.594,13.495,
13.796,11.481,12.49,15.546,11.101,12.577,9.78,15.957,15.711,9.598,13.701,13.404,16.688,18.076,15.516,16.632,13.562,
12.659,15.373,13.359,13.628,14.081,14.607]

ftt_prediction = [12.359,14.432,20.319,12.032,10.918,15.401,12.671,14.752,11.756,14.36,13.5,9.778,13.119,12.277,
14.966,11.804,15.433,20.707,15.332,15.852,14.455,11.299,15.799,15.876,14.581,15.534,14.066,16.38,19.173,11.922,
10.357,11.428,11.444,12.12,13.759,10.939,16.566,19.978,10.735,16.065,16.578,18.134,20.764,9.597,17.031,14.281,
14.096,12.714,10.625,13.919,15.279,14.606,12.847,14.683,15.164,17.879,12.689,20.267,11.83,14.67,15.084,17.183,
13.429,11.032,13.398,11.974,12.954,12.917,14.713,12.326,12.012,13.015,16.294,17.829,7.313,12.329,11.806,16.261,
20.895,14.655,14.327,14.548,12.424,15.408,13.214,13.672,16.557, 14.705]

mlp_prediction = np.array(mlp_prediction, dtype=np.float64)
ftt_prediction = np.array(ftt_prediction, dtype=np.float64)

plt.figure(figsize=(20, 12))
plt.subplot(2, 3, 1)
curve_fitting(rf_prediction, 'Random Forest')
plt.subplot(2, 3, 2)
curve_fitting(xgb_prediction, 'Xgboost')
plt.subplot(2, 3, 3)
curve_fitting(lasso_prediction, 'Lasso Net')
plt.subplot(2, 3, 4)
curve_fitting(ridge_prediction, 'Ridge')
plt.subplot(2, 3, 5)
curve_fitting(mlp_prediction, 'MLP')
plt.subplot(2, 3, 6)
curve_fitting(ftt_prediction, 'FT-Transformer')
plt.show()

feature_name = ['Blood group','Height','Weight','BMI','FFM','Age','Dose/IU','IU/Kg','vVWF:Ag','Pre','t1/h','pk1',
                't2/h','pk2','t3/h','pk3']
xgb.get_booster().feature_names = feature_name
fig, ax = plt.subplots(figsize=(14,10))
ax.tick_params(axis='x', labelsize=16)
ax.tick_params(axis='y', labelsize=16)
xlabel_font = {
    # 'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
    'fontsize': 18,
    # 'fontweight': rcParams['axes.titleweight'], # 设置成和轴刻度标签一样的粗细
    'fontweight': 'light',
    # 'color': rcParams['axes.titlecolor'], # 设置成和轴刻度标签一样的颜色
    'color': 'black',
}
ylabel_font = {
    # 'fontsize': rcParams['axes.titlesize'], # 设置成和轴刻度标签一样的大小
    'fontsize': 18,
    # 'fontweight': rcParams['axes.titleweight'], # 设置成和轴刻度标签一样的粗细
    'fontweight': 'light',
    # 'color': rcParams['axes.titlecolor'], # 设置成和轴刻度标签一样的颜色
    'color': 'black',
}
label_fontdict = {
    'fontsize': 18,
}
ax.set_title('Feature Importance', fontdict=label_fontdict)
ax.set_xlabel('x', fontdict=xlabel_font)
ax.set_ylabel('y', fontdict=ylabel_font)
plot_importance(xgb, ax=ax, height=0.4)
plt.title('Feature Importance', fontdict=label_fontdict)
pyplot.show()

# dump_list = xgb.get_booster().get_dump()
# print(dump_list)
# print(len(dump_list))
xgb.get_booster().feature_names = feature_name
_, ax = plt.subplots(figsize=(10, 10))
plot_tree(xgb, ax=ax, num_trees=xgb.get_booster().best_iteration)
plt.show()

importance = rf.feature_importances_
print("Features sorted by their score:")
print(sorted(zip(map(lambda x: round(x, 4), importance), feature_name), reverse=True))
feature_importances_df = pd.DataFrame({'feature': feature_name, 'importance': importance})
# 对特征重要性得分进行排序
feature_importances_df = feature_importances_df.sort_values('importance', ascending=False)

# 颜色映射
colors = plt.cm.viridis(np.linspace(0, 1, len(feature_name)))

# 可视化特征重要性
_, ax = plt.subplots(figsize=(10, 6))
ax.barh(feature_importances_df['feature'], feature_importances_df['importance'], color=colors)
ax.invert_yaxis()  # 翻转y轴，使得最大的特征在最上面
ax.set_xlabel('feature_importance', fontsize=12)  # 图形的x标签
ax.set_title('feature importance by random forest regressor ', fontsize=16)
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
plt.show()'''


pass
