#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/3/1 17:15
# @Author  : Liangliang
# @File    : featureselection.py
# @Software: PyCharm

import argparse
import datetime
import time
import os
from xmlrpc.client import boolean
os.system("pip install lightgbm")
os.system("pip install xgboost")


import s3fs
import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from xgboost import XGBRegressor
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMClassifier
from lightgbm import LGBMRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score


#读取文件系统
class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


def read_data(file):
    #注意输入的数据不含id，只有feature与label，label在最后一列,且特征的维数要大于1
    data = pd.read_csv("s3://" + file, sep=',', header=None)
    label = data.iloc[:, -1].values
    data = data.iloc[:, :-1].values
    return data, label


def write(data, args):
    #data为一个list类型
    start = time.time()
    n = len(data)  # 数据的数量
    with open(os.path.join(args.data_output, 'pred.csv'), mode="a") as resultfile:
        if n > 1:  # 说明此时的data是[[],[],...]的二级list形式
            for i in range(n):
                line = ",".join(map(str, data[i])) + "\n"
                resultfile.write(line)
        else:  # 说明此时的data是[x,x,...]的list形式
            line = ",".join(map(str, data)) + "\n"
            resultfile.write(line)
    cost = time.time() - start
    print("write is finish. write {} lines with {:.2f}s".format(n, cost))


def feature_select(args):
    # 读取数据
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    #主要是为了知道数据的维数
    data_temp, _ = read_data(input_files[0])
    dim = data_temp.shape[1]
    result = np.zeros((dim, 2)).astype("str")
    del data_temp
    if args.model == "xgboost":
        model = XGBClassifier(n_estimators=args.n_estimators, importance_type=args.importance_type)
    elif args.model == "xgboost_regressor":
        model = XGBRegressor(n_estimators=args.n_estimators, importance_type=args.importance_type)
    elif args.model == "randomforest":
        model = RandomForestClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                       class_weight=args.class_weight)
    elif args.model == "randomforest_regressor":
        model = RandomForestRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth,
                                      class_weight=args.class_weight)
    elif args.model == "lightgbm":
        model = LGBMClassifier(n_estimators=args.n_estimators, max_depth=args.max_depth,
                               class_weight=args.class_weight)
    else:
        model = LGBMRegressor(n_estimators=args.n_estimators, max_depth=args.max_depth,
                              class_weight=args.class_weight)
    test_auc = []
    for file in input_files:
        print("开始读取第{}个文件 {}".format(count, datetime.datetime.now()))
        data, label = read_data(file)
        #将数据划分为训练集和测试集
        train_data, test_data, train_label, test_label = train_test_split(data, label, test_size=args.test_size)
        model.fit(train_data, train_label)
        y_test = model.predict(test_data)
        auc = roc_auc_score(test_label, y_test)
        test_auc.append(auc)
        count = count + 1
    res = model.feature_importances_

    #归一化操作
    if args.model == "lightgbm" or args.model == "lightgbm_regressor":
        if args.is_normalization:
            res = res / res.sum()

    for i in range(dim):
        result[i, 0] = str(i + 1)
        result[i, 1] = str(res[i])
    #写入数据
    write(result.tolist(), args)
    if args.show_auc == "max":
        print("当前模型的AUC值为:", max(test_auc))
    elif args.show_auc == "min":
        print("当前的模型AUC值为:", min(test_auc))
    elif args.show_auc == "mean":
        print("当前的模型AUC值为:", sum(test_auc)/len(test_auc))
    else:
        print("当前的模型AUC值为:", test_auc)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--model", help="特征选择模型类型[xgboost, xgboost_regressor, randomforest,randomforest_regressor,"
                                        " lightgbm, lightgbm_regressor]", type=str, default="lightgbm")
    parser.add_argument("--importance_type", help="xgboost特征重要性类型[weight,gain,cover,total_gain,total_cover]",
                        type=str, default="cover")
    parser.add_argument("--test_size", help="整个数据划分为测试集的占比", type=float, default=0.2)
    parser.add_argument("--n_estimators", help="迭代次数", type=int, default=100)
    parser.add_argument("--is_normalization", help="是否需要对输出的打分进行归一化", type=bool, default=True)
    parser.add_argument("--class_weight", help="类标签权重方式", type=str, default='balanced')
    parser.add_argument("--max_depth", help="决策树的深度", type=int, default=10)
    parser.add_argument("--show_auc", help="最终AUC的展示方式", type=str, default='max')
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')

    args = parser.parse_args()
    feature_select(args)
