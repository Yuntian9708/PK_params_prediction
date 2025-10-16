import math
import random
import rtdl
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import torch
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer
from torch import nn
from torch.utils.data import DataLoader
from tools import set_seed
from tqdm import tqdm
from tools import get_TensorData, get_KFold_data

seed = 42
set_seed(seed)
# pk_path = './data/FVIII_hl_ivr.xlsx'
# pk_path = './data/FVIII_hl_ivr_324.xlsx'
pk_path = './data/FVIII_hl_ivr_24.xlsx'
save_path = './'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
df_pk = pd.read_excel(pk_path)
# df_extrapolation = pd.read_excel(extrapolation_path)
# features = df_pk.iloc[:, 1:17].values  # 三个点
# features = df_pk.iloc[:, 1:15].values  # 两个点
features = df_pk.iloc[:, 1:13].values  # 一个点
hl_label = df_pk.iloc[:, -2].values
hl_label = hl_label.astype('float')

std_scaler = StandardScaler()
minmax_scaler = MinMaxScaler()

norm_features = minmax_scaler.fit_transform(features)
norm_hl = std_scaler.fit_transform(hl_label.reshape(-1, 1))


def train_model(model, x_train, y_train, epochs, device):
    global iterations
    train_loader = DataLoader(torch.utils.data.TensorDataset(x_train, y_train), batch_size=BATCHSIZE, shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    epoch_train_loss = 0.0
    total_optimize_step = math.ceil(len(x_train) / BATCHSIZE) * MAX_EPOCHS
    # early_stopping = EarlyStopping(save_path)

    with tqdm(total=total_optimize_step) as progress_bar:
        for epoch in range(epochs):
            for iterations, (batch_x, batch_y) in enumerate(train_loader):
                model.train().to(device)
                optimizer.zero_grad()
                out = model(batch_x, x_cat=None)
                batch_train_loss = loss(out, batch_y)
                epoch_train_loss += batch_train_loss

                batch_train_loss.backward()  # 反向传播
                optimizer.step()  # 使用梯度进行优化
                progress_bar.update(1)
            epoch_train_loss = epoch_train_loss / iterations  # 计算每一轮迭代的平均损失作为epoch loss


        progress_bar.set_postfix({
            'train loss': '%.4f' % epoch_train_loss,
        })

        # print('training: epoch:{}, train loss is:{:.3f}, test loss is:{:.3f}'.
        #       format(epoch + 1, epoch_train_loss, epoch_test_loss))

        # 早停止
        # early_stopping(epoch_test_loss, model)
        # # # 达到早停止条件时，early_stop会被置为True
        # if early_stopping.early_stop:
        #     print("Early stopping! Best model are saved.")
        #     break  # 跳出迭代，结束训练

    return epoch_train_loss, model


def eval_model(model, x_test, y_test):
    model.eval().to(device)
    pk_output = model(x_test, x_cat=None)
    pk_output = pk_output.cpu().detach().numpy()
    x_test = x_test.cpu().detach().numpy()
    y_test = y_test.cpu().detach().numpy()
    y_test_origin = std_scaler.inverse_transform(y_test)
    pk_output_origin = std_scaler.inverse_transform(pk_output)
    mse = mean_squared_error(y_test_origin, pk_output_origin)
    mae = mean_absolute_error(y_test_origin, pk_output_origin)
    rmse = np.sqrt(mse)
    mape = mean_absolute_percentage_error(y_test_origin, pk_output_origin)
    r2 = r2_score(y_test_origin, pk_output_origin)

    # result = np.squeeze([y_test_origin, pk_output_origin])
    # result_column = ['真实值', '预测值']
    # df = pd.DataFrame(np.array(result)).T
    # df.columns = result_column
    # print('-' * 5, '预测结果', '-' * 5)
    #
    # print(df)
    # print('-' * 5)
    # print('MAE：', round(mae, 3))
    # print('RMSE：', round(rmse, 3))
    # print('MAPE：', round(mape, 3))
    # print('R2 score：', round(r2, 3))

    return mae, rmse, mape, r2


def kfold_val(model,fold):
    rmse_list = []
    mae_list = []
    mape_list = []
    r2_list = []
    fold_idx = 1
    kf = KFold(n_splits=fold, shuffle=True, random_state=seed)

    for train_index, test_index in kf.split(norm_features):
        # print('-' * 5, 'Fold:', fold_idx, '-' * 5)
        # print('Test idx:', test_index+1)
        X_train, y_train = norm_features[train_index], norm_hl[train_index]
        X_test, y_test = norm_features[test_index], norm_hl[test_index]
        X_train, y_train = get_TensorData(X_train, y_train, device)
        X_test, y_test = get_TensorData(X_test, y_test, device)
        train_loss, model = train_model(model, X_train, y_train, MAX_EPOCHS, device)
        fold_mae, fold_rmse, fold_mape, fold_r2 = eval_model(model, X_test, y_test)
        mae_list.append(fold_mae)
        rmse_list.append(fold_rmse)
        mape_list.append(fold_mape)
        r2_list.append(fold_r2)

        fold_idx += 1

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


if __name__ == '__main__':
    seed = 2026
    set_seed(seed)
    fold = 5

    # model configurations
    d_token = 96  # embedding size [96, 128, 192, 256, 320, 384] 192
    n_blocks = 2  # 2,4,6
    attention_dropout = 0.3  # 0.4, 0.5
    ffn_factor = 6 / 3  # 2/3, 4/3, 6/3, 8/3
    ffn_d_hidden = int(d_token * ffn_factor)
    ffn_dropout = 0.0  # 0.1, 0.05
    residual_dropout = 0.0

    # 训练参数
    BATCHSIZE = 8  #
    MAX_EPOCHS = 50
    lr = 1e-4
    weight_decay = 1e-4

    model = rtdl.FTTransformer.make_baseline(
        n_num_features=12,  # blood group and timst to predict are categorical features
        cat_cardinalities=None,
        d_token=d_token,  # embedding size [96, 128, 192, 256, 320, 384]
        n_blocks=n_blocks,
        attention_dropout=attention_dropout,
        ffn_d_hidden=ffn_d_hidden,  # ffn_d_hidden = ffn_factor* d_token, factor=2/3~8/3
        ffn_dropout=ffn_dropout,
        residual_dropout=residual_dropout,
        d_out=1
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    loss = nn.MSELoss()

    kfold_val(model, fold=fold)




