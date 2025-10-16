# encoding=utf-8

import os
import sys

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

sys.path.append(".")
import random
import logging
from sklearn.metrics import precision_score, recall_score, accuracy_score, f1_score, mean_squared_error, auc, \
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay, mean_absolute_error, mean_absolute_percentage_error, \
    r2_score
from typing import Optional, Dict
import pandas as pd
from tqdm import tqdm
from dataclasses import dataclass, field
import numpy as np
import torch
from torch import nn
import transformers
from transformers import (
    HfArgumentParser, set_seed, TrainingArguments, AutoConfig, AutoTokenizer,
    AutoModel,
)
from utils_tabm.average_meter import AverageMeter
from utils_tabm.create_optimizer_scheduler import create_optimizer_and_scheduler
import time
import math
import json
from sklearn import preprocessing
from prettytable import PrettyTable

logger = logging.getLogger(__name__)


class CustomModel(nn.Module):
    def __init__(self, data_args, use_reg=False, cls_classes=None):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            data_args.model_name_or_path
        )
        self.use_reg = use_reg

        hidden_size = self.model.config.hidden_size
        if self.use_reg:
            self.reg_proj = nn.Linear(hidden_size, 1)
        else:
            self.cls_proj = nn.Linear(hidden_size, cls_classes)

    def forward(self,
                **inputs
                ):
        scores = self.model(**inputs, return_dict=True).last_hidden_state
        scores = scores[:, 0, :]
        if self.use_reg:
            return self.reg_proj(scores)
        else:
            return self.cls_proj(scores)


def normalize(X: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    X_train = X.copy()
    normalizer = preprocessing.MinMaxScaler()
    # normalizer = preprocessing.StandardScaler()
    normalizer.fit(X_train)
    return normalizer


def count_parameters(model, param_table=False):
    table = PrettyTable(["Modules", "Parameters"])
    total_trainable_params, total_params = 0, 0
    for name, parameter in model.named_parameters():
        param = parameter.numel()
        total_params += param
        if parameter.requires_grad:
            table.add_row([name, param])
            total_trainable_params += param

    if param_table:
        print(table)
    selflogger.info(f"Total Trainable Params: {total_trainable_params}, {round(total_trainable_params / 1e06, 2)} M")
    selflogger.info(f"Total Params: {total_params}, {round(total_params / 1e06, 2)} M")
    return total_trainable_params, total_params


def get_training_dataset(path, indices, use_reg=False):
    csv_data = None
    if path.split(".")[-1] == "csv":
        csv_data = pd.read_csv(path, encoding='gbk')

    columns = list(csv_data.columns)
    columns = [_ for _ in columns if "Unnamed:" not in _]

    x_columns = columns[:-1]
    x_columns = [_ for _ in x_columns if len(_) > 0]

    y_column = columns[-1]
    x_inputs, y_labels = [], []

    y_labels_norm = np.array([[_] for _ in csv_data[y_column]])

    if use_reg:
        # normalize y_labels
        # std_scaler = StandardScaler()
        normalizer = normalize(y_labels_norm)
        y_labels_norm = normalizer.transform(y_labels_norm)
        # y_labels_norm = std_scaler.fit_transform(y_labels_norm)
        y_labels_norm = [_[0] for _ in y_labels_norm]

    for i in indices:
        features = csv_data.iloc[i]
        # y_feature = features[y_column]
        y_feature = y_labels_norm[i]
        x_features = features[x_columns]

        x_features = list(x_features)

        x_inputs.append(
            " [SEP] ".join([f"{x_columns[idx]} is {_}" for idx, _ in enumerate(x_features)])
        )
        if not use_reg:
            y_feature = int(y_feature)

        y_labels.append(y_feature)

    cls_classes = None
    if not use_reg:
        cls_classes = len(set(list(csv_data[y_column])))
    return x_inputs, y_labels, cls_classes


@dataclass
class DataTrainingArguments:
    load_pretrained: Optional[bool] = field(default=True)
    pretrained_model: Optional[str] = field(default=None)
    all_datasets_path: Optional[str] = field(default="datasets_norm_num")
    train_test_indices: Optional[str] = field(default="indices/train_test_indices_reg.json")
    use_reg: Optional[bool] = field(default=False)
    cache_dir: Optional[bool] = field(default="cache_dir")
    config_name: Optional[bool] = field(default="bge-reranker")
    model_name_or_path: Optional[bool] = field(default="bge-reranker")
    lora_r: Optional[int] = field(default=16)
    lora_alpha: Optional[int] = field(default=32)
    lora_dropout: Optional[float] = field(default=0.1)
    load_in_8bit: Optional[bool] = field(default=True)
    use_fast_tokenizer: Optional[bool] = field(default=True)
    tokenizer_name: Optional[str] = field(default=None)


if __name__ == "__main__":
    torch.cuda.empty_cache()
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    to_cuda = True
    selflogger = logging.getLogger('THIS-LOGGING')
    selflogger.setLevel(logging.INFO)
    handler = logging.FileHandler(f"logs-lm-tab/log-{time.time()}.log", encoding='utf-8')
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    selflogger.addHandler(console_handler)
    selflogger.addHandler(handler)

    # device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
    # to_cuda = True
    # device = torch.device("cpu")
    # to_cuda = False

    parser = HfArgumentParser((TrainingArguments, DataTrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
        training_args, data_args = training_args
    else:
        training_args = parser.parse_args_into_dataclasses()

    selflogger.info(training_args)
    selflogger.info(data_args)
    set_seed(training_args.seed)

    # Setup logging
    logging.basicConfig(format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", datefmt="%m/%d/%Y %H:%M:%S",
                        handlers=[logging.StreamHandler(sys.stdout)], )
    log_level = training_args.get_process_log_level()

    dataset_loglevel, transformers_loglevel = 20, 30

    logger.setLevel(log_level)
    transformers.utils.logging.set_verbosity(transformers_loglevel)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()

    ################
    use_reg = data_args.use_reg
    selflogger.info(f"use_reg {use_reg}")
    ################
    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)
    ################
    all_datasets_path = data_args.all_datasets_path
    all_datasets_path = [os.path.join(all_datasets_path, _) for _ in os.listdir(all_datasets_path)]
    all_datasets_path = sorted(all_datasets_path)

    num_datasets = len(all_datasets_path)

    selflogger.info(f"all_datasets_path {all_datasets_path}, count = {len(all_datasets_path)}")
    print('data_path:', all_datasets_path)

    if not os.path.exists(training_args.output_dir):
        os.mkdir(training_args.output_dir)
    ################
    eval_log_step = 100
    selflogger.info(f"eval_log_step {eval_log_step}")
    ################
    config = AutoConfig.from_pretrained(
        data_args.config_name if data_args.config_name else data_args.model_name_or_path,
        cache_dir=data_args.cache_dir,
    )
    ################

    tokenizer = AutoTokenizer.from_pretrained(
        data_args.tokenizer_name if data_args.tokenizer_name else data_args.model_name_or_path,
        cache_dir=data_args.cache_dir,
        use_fast=data_args.use_fast_tokenizer,
    )
    selflogger.info(f"tokenizer.pad_token_id {tokenizer.pad_token_id}")
    selflogger.info(config)

    with open(data_args.train_test_indices, "r", encoding="utf-8") as r_f:
        train_test_indices = json.load(r_f)

    train_ratio = 0.8

    for dataset_i, current_dataset in enumerate(all_datasets_path):

        if "FVIII_half_life_processed_324" not in current_dataset:
            print("Half life Dataset not found")
            exit(0)
        print('Current Dataset', current_dataset)

        csv_data = pd.read_csv(current_dataset, encoding="gbk")

        all_indices = list(range(len(csv_data)))
        train_size = round(train_ratio * len(all_indices))
        train_indices, test_indices = None, None

        # with open("indices/itp_train_test_idx.json", "r", encoding="utf-8") as r_f:
        #     train_test_indices = json.load(r_f)
        #     train_indices = train_test_indices["train_idx"]
        #     test_indices = train_test_indices["test_idx"]
        acc_list = []
        pre_list = []
        recall_list = []
        f1_list = []
        auc_list = []

        mae_list = []
        rmse_list = []
        r2_list = []

        for random_i in range(20):
            # for random_i in range(1):

            selflogger.info(f"random_i = {random_i}")

            random.shuffle(all_indices)
            train_indices, test_indices = all_indices[:train_size], all_indices[train_size:]

            train_x_inputs, train_y_labels, cls_classes = get_training_dataset(current_dataset, train_indices,
                                                                               use_reg=use_reg)

            total_train_size = len(train_x_inputs)

            selflogger.info(f"current_dataset {current_dataset}, train {len(train_indices)}, test {len(test_indices)}")

            tabular_lm = CustomModel(data_args=data_args, use_reg=use_reg, cls_classes=cls_classes)
            count_parameters(tabular_lm)

            # 先冻结所有层参数
            for param in tabular_lm.parameters():
                param.requires_grad = False

            # 解冻embedding层
            for param in tabular_lm.model.embeddings.parameters():
                param.requires_grad = True

            # 解冻第一层
            for param in tabular_lm.model.encoder.layer[0].parameters():
                param.requires_grad = True


            # 解冻最后一层
            for param in tabular_lm.model.encoder.layer[-1].parameters():
                param.requires_grad = True

            # 解冻池化层
            # for param in tabular_lm.model.pooler.parameters():
            #     param.requires_grad = True

            # 解冻分类/回归器
            for param in tabular_lm.reg_proj.parameters():
                param.requires_grad = True

            # 验证冻结状态
            # for name, param in tabular_lm.named_parameters():
            #     print(f"{name}: {param.requires_grad}")
            #
            # exit(0)

            tabular_lm.to(device)
            tabular_lm.train()
            ################
            train_batch_size = training_args.per_device_train_batch_size

            total_epoch = 20

            total_optimize_step = math.ceil(total_train_size / train_batch_size) * total_epoch
            selflogger.info(
                f"total_epoch {total_epoch}, total_optimize_step {total_optimize_step}, train_batch_size {train_batch_size}")

            optimizer, lr_scheduler = create_optimizer_and_scheduler(tabular_lm, args=training_args,
                                                                     num_training_steps=total_optimize_step)
            ################

            average_meter = AverageMeter()
            classify_fn = torch.nn.CrossEntropyLoss() if not use_reg else torch.nn.MSELoss()
            # MSE Loss: torch.nn.MSELoss()
            # MAE Loss: torch.nn.L1Loss()
            # Smooth MAE Loss: torch.nn.SmoothL1Loss()
            # Huber Loss: torch.nn.HuberLoss()
            # Log cosh Loss:

            # gradient_accumulation_steps = training_args.gradient_accumulation_steps
            # scaler = torch.cuda.amp.GradScaler()
            with tqdm(total=total_optimize_step) as progress_bar:

                for epoch_i in range(total_epoch):
                    # shuffle with random indices
                    new_indices = np.random.permutation(len(train_y_labels))
                    train_x_inputs = [train_x_inputs[_] for _ in new_indices]
                    train_y_labels = [train_y_labels[_] for _ in new_indices]

                    for batch_i in range(0, total_train_size, train_batch_size):
                        x_inputs = train_x_inputs[batch_i:batch_i + train_batch_size]
                        y_labels = train_y_labels[batch_i:batch_i + train_batch_size]

                        inputs = tokenizer(x_inputs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                        for k, v in inputs.items():
                            inputs[k] = v.to(device)

                        hidden_states = tabular_lm(**inputs)

                        if use_reg:
                            hidden_states = hidden_states.squeeze(1)

                        loss = None
                        optimizer.zero_grad()

                        y_labels = torch.tensor(y_labels, dtype=torch.long,
                                                device=device) if not use_reg else torch.tensor(y_labels,
                                                                                                dtype=torch.float32,
                                                                                                device=device)

                        loss = classify_fn(hidden_states, y_labels)

                        average_meter.update(loss.cpu().item(), hidden_states.size(0))
                        # scaler.scale(loss).backward()
                        # scaler.step(optimizer)
                        # scaler.update()
                        loss.backward()
                        optimizer.step()
                        lr_scheduler.step()
                        progress_bar.update(1)
                progress_bar.close()

            selflogger.info(f"train loss {average_meter.avg}")

            selflogger.info(f" === evaluating ===")
            tabular_lm.eval()


            average_meter = AverageMeter()
            preditions = []

            test_x_inputs, test_y_labels, cls_classes = get_training_dataset(current_dataset, test_indices,
                                                                             use_reg=use_reg)
            total_test_size = len(test_y_labels)

            with torch.no_grad():
                with tqdm(total=total_test_size) as progress_bar:
                    for batch_i in range(0, total_test_size, train_batch_size):

                        x_inputs = test_x_inputs[batch_i:batch_i + train_batch_size]
                        y_labels = test_y_labels[batch_i:batch_i + train_batch_size]

                        inputs = tokenizer(x_inputs, padding=True, truncation=True, return_tensors='pt', max_length=512)
                        for k, v in inputs.items():
                            inputs[k] = v.to(device)

                        hidden_states = tabular_lm(**inputs)

                        if use_reg:
                            hidden_states = hidden_states.squeeze(1)

                        y_labels = torch.tensor(y_labels, dtype=torch.long,
                                                device=device) if not use_reg else torch.tensor(y_labels,
                                                                                                dtype=torch.float32,
                                                                                                device=device)
                        loss = classify_fn(hidden_states, y_labels)
                        average_meter.update(loss.cpu().item(), hidden_states.size(0))

                        if not use_reg:
                            hidden_states = torch.argmax(hidden_states, dim=-1)

                        hidden_states = hidden_states.cpu().tolist()
                        preditions.extend(hidden_states)

                        progress_bar.update(train_batch_size)

                    progress_bar.close()

                    if use_reg:
                        mae_result = mean_absolute_error(test_y_labels, preditions)
                        rmse_result = mean_squared_error(test_y_labels, preditions) ** 0.5
                        r2_result = r2_score(test_y_labels, preditions)
                        if r2_result < 0:
                            r2_result += 1


                        # selflogger.info(f"eval loss {average_meter.avg}")
                        selflogger.info(f"mae = {mae_result:.4f}, rmse = {rmse_result:.4f}, r2 = {r2_result:.4f}")
                        selflogger.info(f"\n")
                        # mae_list.append(mae_result)
                        # rmse_list.append(rmse_result)
                        # r2_list.append(r2_result)

                    else:
                        accuracy = round(accuracy_score(test_y_labels, preditions), 4)
                        recall = round(recall_score(test_y_labels, preditions), 4)
                        precision = round(precision_score(test_y_labels, preditions), 4)
                        f1score = round(f1_score(test_y_labels, preditions), 4)
                        # auc = round(roc_auc_score(test_y_labels, preditions), 4)
                        cm = confusion_matrix(test_y_labels, preditions)
                        # cm_plot = ConfusionMatrixDisplay(cm).plot()
                        # plt.show()

                        # selflogger.info(f"eval loss {average_meter.avg}")
                        selflogger.info(f"accu = {accuracy}, prec = {precision}, reca = {recall}, f1score = {f1score}, "
                                        f"cm = {cm}")
                        selflogger.info(f"\n")
                        # acc_list.append(accuracy)
                        # pre_list.append(precision)
                        # recall_list.append(recall)
                        # f1_list.append(f1score)

                        # auc_list.append(auc)
        # if use_reg:
        #     selflogger.info(f"avg_mae = {np.mean(mae_list):.4f}, avg_rmse = {np.mean(rmse_list):.4f}, "
        #                     f"avg_r2 = {np.mean(r2_list):.4f}")
        # else:
        #     selflogger.info(f"avg_acc = {np.mean(acc_list)}, avg_prec = {np.mean(pre_list)}, "
        #                     f"avg_reca = {np.mean(recall_list)}, avg_f1score = {np.mean(f1_list)}, ")
    selflogger.info(f"=== done ===")
