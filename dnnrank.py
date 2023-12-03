#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/2/22 15:02
# @Author  : Liangliang
# @File    : dnnrank.py
# @Software: PyCharm

import os
import datetime
import time
import argparse
import s3fs

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras as keras


#读取文件系统
class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


#定义精排DNN模型结构
class Dnn(keras.Model):
    def __init__(self, layer_dim1 = 200, layer_dim2 = 100, output_dim = 2):
        super(Dnn, self).__init__()
        self.cov1 = keras.layers.Dense(layer_dim1, use_bias=True)
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.cov2 = keras.layers.Dense(layer_dim2, use_bias=True)
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.cov3 = keras.layers.Dense(output_dim, use_bias=True)

    def call(self, inputs, training=None, mask=None):
        h = self.cov1(inputs)
        #if training:
        #    h = self.batch_normalization1(h, training=training)
        h = tf.nn.relu(h)
        h = self.cov2(h)
        #if training:
        #    h = self.batch_normalization2(h, training=training)
        h = tf.nn.relu(h)
        h = self.cov3(h)
        #h = tf.nn.leaky_relu(h)
        h = tf.nn.sigmoid(h)
        h = tf.nn.softmax(h, axis=1)
        return h


#计算loss函数
def loss_function(predict, label, args):
    #防止log函数中输入值为0
    e = 1e-06
    #样本的权重 未点击:权重为1 电点击未成功加入:beta 点击成功加入:gamma  beta<gamma均为大于1的参数
    # 赋予不同样本不同的权值
    weight = tf.where(tf.equal(label, 1), args.gamma, label)
    weight = tf.where(tf.equal(weight, 0), args.beta, weight)
    weight = tf.where(tf.equal(weight, -1), 1.0, weight)
    label = tf.nn.relu(label)
    n = label.shape[0]
    # 预测结果转换成矩阵
    pred1 = predict[:, 0]
    pred2 = 1 - pred1
    #数据截断防止loss出现nan的情况
    pred1 = tf.reshape(tf.where(tf.less(pred1, e), e, pred1), [-1, 1])
    pred2 = tf.reshape(tf.where(tf.less(pred2, e), e, pred2), [-1, 1])

    #计算交叉熵
    #loss_cr = -tf.reduce_sum(1 / n*(label * tf.math.log(pred1) + (1 - label) * tf.math.log(pred2)) * weight)
    loss_cr = tf.reduce_sum(tf.nn.weighted_cross_entropy_with_logits(label, pred2, pos_weight=args.gamma))

    #正样本的预测概率
    poss_pred = (1 - label) * pred1 + label * pred2
    #负样本的预测概率
    neg_pred = 1 - poss_pred
    #bpr损失是拉大正样本与负样本之间的预测概率
    loss_bpr = -tf.reduce_sum(1 / n * tf.math.log(tf.sigmoid(poss_pred - neg_pred)) * weight)
    loss = (loss_cr / args.gamma + args.alpha * loss_bpr)/2
    return loss


#读取数据
def read_data(args, epoch, count, file):
    if args.env == "train" or args.env == "train_incremental":
        print("开始读取第{}个epoch的第{}个文件数据! {}".format(epoch, count, datetime.datetime.now()))
    elif args.env == "test":
        print("开始读取第{}个文件数据! {}".format(count, datetime.datetime.now()))
    data = pd.read_csv("s3://" + file, sep=',', header=None).astype("str")
    #训练环境读取数据  data最后一列为label
    if args.env == "train" or args.env == "train_incremental":
        data = tf.convert_to_tensor(data.iloc[:,2::].values, dtype=tf.float32)
        return data
    #预测环境读取数据 返回: id[玩家id,战队id], 数据data
    else:
        id = data.iloc[:, 0:2]
        data = tf.convert_to_tensor(data.iloc[:, 2::].values, dtype=tf.float32)
        return id, data


def write(data, count, args):
    #data为一个list类型
    start = time.time()
    n = len(data)  # 数据的数量
    with open(os.path.join(args.data_output, 'pred_{}.csv'.format(count)), mode="a") as resultfile:
        if n > 1:  # 说明此时的data是[[],[],...]的二级list形式
            for i in range(n):
                line = ",".join(map(str, data[i])) + "\n"
                resultfile.write(line)
        else:  # 说明此时的data是[x,x,...]的list形式
            line = ",".join(map(str, data)) + "\n"
            resultfile.write(line)
    cost = time.time() - start
    print("write is finish. write {} lines with {:.2f}s".format(n, cost))


def train(args):
    if args.env == "train":
        # 需要训练一个新模型
        dnn_model = Dnn(args.input_dim, args.feat_dim, args.output_dim)
    else:
        # 利用之前的模型采样增量式的方式进行训练
        # 装载训练好的模型
        cmd = "s3cmd get -r  " + args.model_output + "dnn"
        os.system(cmd)
        dnn_model = keras.models.load_model("./dnn", custom_objects={'tf': tf}, compile=False)
        print("Model is loaded!")

    #读取验证集数据
    path = args.data_input.split(',')[1]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    validation_data = pd.DataFrame()
    for file in input_files:
        data = pd.read_csv("s3://" + file, sep=',', header=None).astype("str")
        validation_data = pd.concat([validation_data, data], axis=0)

    #读取验证集的数据与类标签
    validation_data = tf.convert_to_tensor(validation_data.iloc[:, 2::].values, dtype=tf.float32)
    validation_label = tf.reshape(validation_data[:, -1], [-1, 1])
    validation_data = validation_data[:, :-1]

    #读取训练集数据
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    # 定义优化器
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)

    stop_num = 0
    loss = 0

    #strategy = tf.distribute.MirroredStrategy()
    #print("gpu的数量为:{}".format(strategy.num_replicas_in_sync))
    roc = 0

    for epoch in range(args.epoch):
        count = 0
        for file in input_files:
            #读取数据
            data = read_data(args, epoch, count, file)
            count = count + 1
            label = tf.reshape(data[:,-1],[-1,1])
            #排除label那一列防止出现过拟合
            data = data[:, :-1]

            #对输入数据划分batch
            data = tf.data.Dataset.from_tensor_slices((data, label)).shuffle(100).batch(args.batch_size,
                                                                                        drop_remainder=True)
            #分发数据
            #data = strategy.experimental_distribute_dataset(data)

            #with strategy.scope():
            for batch_data, batch_label in data:
                with tf.GradientTape(persistent=True) as tape:
                    predict = dnn_model(batch_data, training=True)
                    loss = loss_function(predict, batch_label, args)
                gradients = tape.gradient(loss, dnn_model.trainable_variables)
                optimizer.apply_gradients(zip(gradients, dnn_model.trainable_variables))
            # 评估算法效果
            print("第{}个epoch第{}个文件的训练loss为:{}".format(epoch, count, loss))
            predict_val = dnn_model(validation_data, training=True)
            predict_label = tf.reshape(predict_val[:, 1], [-1, 1])
            print("真实的正样本类标签比例:", tf.reduce_sum(tf.nn.relu(validation_label)) / validation_label.shape[0])
            print("预测的正样本类标签比例:", tf.reduce_sum(tf.where(tf.less(predict_label, 0.5), 0, 1)) /
                  predict_label.shape[0])
            print("第{}个epoch第{}个文件的roc_score为:{} 准确率为:{}".format(epoch, count,
                                                                               metrics.roc_auc_score(tf.nn.relu(
                                                                                   validation_label), predict_label),
                                                                               metrics.accuracy_score(
                                                                                 tf.nn.relu(validation_label),
                                                                                 tf.where(tf.less(predict_label, 0.5),
                                                                                          0, 1))))

        #检查是否需要保存模型
        roc_val = metrics.roc_auc_score(tf.nn.relu(validation_label), predict_label)
        if roc_val > roc:
            roc = roc_val
            stop_num = 0
            # 保存神经网络模型
            dnn_model.save("./dnn", save_format="tf")
            cmd = "s3cmd put -r ./dnn " + args.model_output
            os.system(cmd)
            print("dnn_model已保存!")
        else:
            stop_num = stop_num + 1
            if stop_num >= args.stop_num:
                print("Early stop!")
                break



def test(args):
    # 装载训练好的模型
    cmd = "s3cmd get -r  " + args.model_output + "dnn"
    os.system(cmd)
    dnn_model = keras.models.load_model("./dnn", custom_objects={'tf': tf}, compile=False)
    print("Model is loaded!")
    # 读取数据
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    for file in input_files:
        # 读取数据 id为pandas.dataframe类型  data为tensor(tf.float32型)
        id, data = read_data(args, 0, count, file)
        #第0列为玩家id, 第1列为clubid, 第2列为score
        result = np.zeros((data.shape[0], 3)).astype("str")
        #写入id
        result[:, 0:2] = id.values
        predict = dnn_model(data)
        result[:, 2] = predict.numpy()[:, 1].astype("str")
        print("第{}个文件数据预测完成! {}".format(count, datetime.datetime.now()))
        #写入结果
        write(result.tolist(), count, args)
        print("第{}个文件数据预测结果写入完成! {}".format(count, datetime.datetime.now()))
        count = count + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or test)", type=str, default='train_incremental')
    parser.add_argument("--lr", help="学习率", type=float, default=0.0001)
    parser.add_argument("--alpha", help="bpr loss权重", type=float, default=35)
    parser.add_argument("--beta", help="点击未加入样本权重", type=float, default=2)
    parser.add_argument("--gamma", help="点击且成功加入样本权重", type=float, default=35)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=30)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=200)
    parser.add_argument("--batch_size", help="batch的大小", type=int, default=3000)
    parser.add_argument("--input_dim", help="输入特征的维度", type=int, default=200)
    parser.add_argument("--feat_dim", help="隐含层神经元的数目", type=int, default=100)
    parser.add_argument("--output_dim", help="输出特征的维度", type=int, default=2)
    parser.add_argument("--nodes_num", help="采样子图的节点数目", type=int, default=1)
    parser.add_argument("--file_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=150000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='s3://models/gclmec/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')

    args = parser.parse_args()
    if args.env == "train" or args.env == "train_incremental":
        train(args)
    elif args.env == "test":
        test(args)
    else:
        print("输入的环境参数错误, env只能为train或test!")
    print("执行完成")