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

seed = 567
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
ivr_label = df_pk.iloc[:, -1].values
ivr_label = ivr_label.astype('float')
# hl_label = df_pk.iloc[:, -2].values
# extrapolation = df_extrapolation.iloc[:, 1:17].values
std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()
qt_scaler = QuantileTransformer()
norm_features = minmax_scaler.fit_transform(features)
norm_ivr = std_scaler.fit_transform(ivr_label.reshape(-1, 1))


# K-Fold split
fold = 5


def kfold_val(fold, model):
    rmse_list = []
    mae_list = []
    mape_list = []
    r2_list = []
    prediction_list = []

    kf = KFold(n_splits=fold)
    for train_index, test_index in kf.split(features):
        # print('Test idx:', test_index)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = ivr_label[train_index], ivr_label[test_index]
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        n = len(ivr_label)
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mape = mean_absolute_percentage_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        # print('Testing results: \n', np.round(y_pred.reshape(-1, 1).transpose(0, 1), 3))
        # print('Fold MAE: {:.3f}'.format(mae))
        # print('Fold RMSE : {:.3f}'.format(rmse))
        # print('Fold MAPE: {:.3f}'.format(mape))
        # print('Fold R2: {:.3f}'.format(r2))

        # scatter_parameter = np.polyfit(y_pred, y_test, 1)
        # print('scatter parameter:', scatter_parameter)
        # y_test2 = scatter_parameter[0] * y_pred + scatter_parameter[1]
        # plt.scatter(y_pred, y_test)
        # plt.plot(y_pred, y_test2, color='g')
        # plt.show()

        rmse_list.append(rmse)
        mae_list.append(mae)
        mape_list.append(mape)
        r2_list.append(r2)
        prediction_list.append(y_pred)

    avg_rmse = np.mean(rmse_list)
    avg_mae = np.mean(mae_list)
    avg_mape = np.mean(mape_list)
    avg_r2 = np.mean(r2_list)

    print('-' * 10)
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

    kf = KFold(n_splits=fold, random_state=seed, shuffle=True)
    for train_index, test_index in kf.split(norm_features):
        # print('Test idx:', test_index+1)
        X_train, X_test = features[train_index], features[test_index]
        y_train, y_test = norm_ivr[train_index], norm_ivr[test_index]
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

    print('-' * 10)
    print('Avg MAE : {:.3f}'.format(avg_mae))
    print('Avg RMSE : {:.3f}'.format(avg_rmse))
    # print('Avg MAPE: {:.3f}'.format(avg_mape))
    print('Avg R2 : {:.3f}'.format(avg_r2))
    # print('Avg Adjusted R2 : {}'.format(avg_adjusted_r2))


def curve_fitting(prediction_list, model_name):
    scatter_parameter = np.polyfit(prediction_list, ivr_label, 1)
    corr,_ = pearsonr(prediction_list, ivr_label)
    print('scatter parameter:', scatter_parameter)
    ivr_label2 = scatter_parameter[0] * prediction_list + scatter_parameter[1]
    plt.scatter(prediction_list, ivr_label)
    plt.plot(prediction_list, ivr_label2, color='g')
    plt.title(model_name, fontsize=14)
    plt.xlabel('Predicted IVR Value', fontsize=14)
    plt.ylabel('Observed IVR Value', fontsize=14)
    plt.text(0.85, 0.05, f'y = {scatter_parameter[0]:.2f}x + {scatter_parameter[1]:.2f}',
             transform=plt.gca().transAxes, fontsize=14, color='black', ha='right')
    plt.text(0.02, 0.92, f'Correlation Coefficient: {corr:.2f}', transform=plt.gca().transAxes, fontsize=12,
             color='black')
    # plt.show()


# RF
rf = RandomForestRegressor(n_estimators=100, max_depth=20, random_state=seed, min_samples_leaf=1)
# XGB
xgb = XGBRegressor(max_depth=20, learning_rate=0.1, n_estimators=100, random_state=seed)
# Lasso
lasso = Lasso(alpha=0.01, random_state=seed)
# Ridge
ridge = Ridge(random_state=seed)
# Bagging
# bagging = BaggingRegressor(n_estimators=100, max_features=9)
# MLP
mlp = MLPRegressor(hidden_layer_sizes=(128, ), activation='relu', solver='lbfgs', learning_rate='adaptive',
                   learning_rate_init=1e-3, shuffle=True, random_state=seed, max_iter=500, batch_size=16)


model_list = [rf, xgb, lasso, ridge, mlp]
name_list = ['Random Forest', 'XGBoost', 'Lasso Net', 'Ridge', 'MLP']

idx = 0
for model in model_list:
    print('**********', name_list[idx], '**********')
    if name_list[idx] == 'MLP':
        kfold_val_norm(fold, model)
    else:
        prediction = kfold_val(fold, model)
        # curve_fitting(prediction, name_list[idx])
    idx += 1


# rf_prediction = kfold_val(fold, rf)
# xgb_prediction = kfold_val(fold, xgb)
# lasso_prediction = kfold_val(fold, lasso)
# ridge_prediction = kfold_val(fold, ridge)

# curve_fitting(rf_prediction, 'Random Forest')
# rf_prediction, rf = kfold_val(fold, rf)
# xgb_prediction, _ = kfold_val(fold, xgb)
# lasso_prediction, lasso = kfold_val(fold, lasso)
# ridge_prediction, ridge = kfold_val(fold, ridge)
# kfold_val_norm(fold, mlp)
'''
mlp_prediction = [1.724,1.787,2.168,1.547,1.607,1.671,1.579,2.188,1.713,1.513,1.733,1.723,1.693,1.648,1.894,
1.407,1.678,2.358,1.626,1.392,1.4,1.721,1.777,1.821,1.693,1.812,1.558,1.992,1.684,1.791,2.008,1.645,2.129,
1.928,1.472,1.896,1.324,2.231,1.704,1.905,1.736,2.538,2.672,1.189,1.96,1.261,2.039,1.967,1.805,1.965,1.736,
1.891,1.677,1.773,2.01,1.402,1.608,1.558,1.621,1.398,1.695,1.807,1.492,2.136,1.657,2.113,1.809,1.459,1.859,1.545,
1.528,1.401,1.635,1.699,1.496,1.639,1.45,1.796,1.898,1.349,1.173,1.132,1.731,1.261,1.828,1.529,1.539,1.368]

ftt_prediction = [1.71899,1.773595,2.185228,1.544324,1.598346,1.649064,1.572458,2.246578,1.736116,1.518121,1.725082,
1.657848,1.708982,1.662512,1.88935,1.318951,1.671474,2.522593,1.63728,1.378107,1.41303,1.712848,1.804642,1.784911,
1.69805,1.836093,1.539601,2.056751,1.705775,1.79497,2.02553,1.595429,2.157645,1.860467,1.475523,1.925655,1.379804,
2.248719,1.747632,1.874352,1.741442,2.595314,2.560407,1.2115,1.95782,1.247655,2.003727,2.050236,1.791685,1.964673,
1.795016,1.902299,1.648658,1.730957,1.990306,1.42682,1.641639,1.563174,1.615354,1.375954,1.673361,1.828917,1.47248,
2.133221,1.617024,2.18119,1.777489,1.438077,1.850435,1.674557,1.509539,1.405083,1.639209,1.680742,1.449484,1.606644,
1.413899,1.791222,1.892612,1.384882,1.12895,1.124548,1.714254,1.19809,1.846739,1.503503,1.53267,1.363754]

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
_, ax = plt.subplots(figsize=(20, 20))
plot_tree(xgb, ax=ax, num_trees=10)
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
plt.show()
'''
pass
