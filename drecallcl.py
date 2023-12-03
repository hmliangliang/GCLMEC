#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2023/2/24 15:31
# @Author  : Liangliang
# @File    : drecallcl.py
# @Software: PyCharm
import os
import time
import datetime
import argparse
import s3fs

import numpy as np
import pandas as pd
from sklearn import metrics
import tensorflow as tf
import tensorflow.keras as keras

#防止除数为0
e = 1e-5

#读取文件系统
class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


#定义双塔模型
class Double_towers(keras.Model):
    def __init__(self, feat_dim1 = 200, feat_dim2 = 150, feat_dim3 = 100, output_dim = 64):
        super(Double_towers, self).__init__()
        self.cov1 = keras.layers.Dense(feat_dim1, use_bias=True)
        self.cov2 = keras.layers.Dense(feat_dim2, use_bias=True)
        self.cov3 = keras.layers.Dense(feat_dim3, use_bias=True)
        self.cov4 = keras.layers.Dense(output_dim)

    def call(self, inputs, training=None, mask=None):
        inputs = self.cov1(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        inputs = self.cov2(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        inputs = self.cov3(inputs)
        inputs = tf.nn.leaky_relu(inputs)
        inputs = self.cov4(inputs)
        return inputs


def loss_function(embed_user, embed_club, label, args):
    #输入不含id
    n = label.shape[0]
    cos_sim = tf.reduce_sum(embed_user * embed_club, axis=1)/((tf.norm(embed_user, ord=2, axis=1)
                                                               *tf.norm(embed_club, ord=2, axis=1)))
    cos_sim = tf.reshape(cos_sim, [-1, 1])
    label = tf.nn.relu(label)
    weight = tf.where(tf.greater(label, 0.5), args.pos_weight, 1.0)
    cos_sim = tf.math.abs(cos_sim - label)
    loss = 1 / n * tf.reduce_sum(tf.math.exp(cos_sim)*weight)
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
    #预测环境读取数据 返回: id[玩家id或战队id], 数据data
    else:
        id = data.iloc[:, 0:2]
        data = tf.convert_to_tensor(data.iloc[:, 2::].values, dtype=tf.float32)
        return id, data


def train(args):
    #读取验证集
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
    validation_label = tf.nn.relu(tf.reshape(validation_data[:, -1], [-1, 1]))
    validation_data = validation_data[:, :-1]

    #判断运行环境
    if args.env == "train":
        # 需要训练一个新模型
        model_user = Double_towers(args.input_dim, args.feat_dim, args.feat_dim2, args.output_dim)
        model_club = Double_towers(args.input_dim, args.feat_dim, args.feat_dim2, args.output_dim)
    else:
        # 利用之前的模型采样增量式的方式进行训练
        # 装载训练好的模型
        cmd = "s3cmd get -r  " + args.model_output
        os.system(cmd)
        model_user = keras.models.load_model("./modeluser", custom_objects={'tf': tf}, compile=False)
        model_club = keras.models.load_model("./modelclub", custom_objects={'tf': tf}, compile=False)
        print("Model is loaded!")

    # 读取训练集数据
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    # 定义优化器
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    before_loss = 2 ** 32
    stop_num = 0
    loss = 0
    roc = 0
    for epoch in range(args.epoch):
        count = 0
        for file in input_files:
            data = read_data(args, epoch, count, file)
            label = tf.reshape(data[:, -1], [-1, 1])
            label = tf.nn.relu(label)
            # 排除label那一列防止出现过拟合
            data = data[:, :-1]

            # 对输入数据划分batch
            data = tf.data.Dataset.from_tensor_slices((data, label)).shuffle(100).batch(args.batch_size,
                                                                                        drop_remainder=True)
            for batch_data, batch_label in data:
                batch_data_user = batch_data[:, 0:args.split_dim]
                batch_data_club = batch_data[:, args.split_dim:]
                with tf.GradientTape(persistent=True) as tape:
                    embed_user = model_user(batch_data_user)
                    embed_club = model_club(batch_data_club)
                    loss = loss_function(embed_user, embed_club, batch_label, args)
                #更新model_user的参数
                gradients = tape.gradient(loss, model_user.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model_user.trainable_variables))

                # 更新model_club的参数
                gradients1 = tape.gradient(loss, model_club.trainable_variables)
                optimizer.apply_gradients(zip(gradients1, model_club.trainable_variables))

            # 评估算法效果
            print("第{}个epoch第{}个文件的训练loss为:{}".format(epoch, count, loss))
            predict_embed_user = model_user(validation_data[:, 0:args.split_dim])
            predict_embed_club = model_club(validation_data[:, args.split_dim:])
            predict_label = tf.reduce_sum(predict_embed_user * predict_embed_club, axis=1)/\
                            (tf.norm(predict_embed_user, ord=2, axis=1)*tf.norm(predict_embed_club,
                                                                                ord=2, axis=1) + e)
            predict_label = tf.reshape(predict_label, [-1, 1])
            print("真实的正样本类标签比例:", tf.reduce_sum(tf.nn.relu(validation_label)) / validation_label.shape[0])
            print("预测的正样本类标签比例:", tf.reduce_sum(tf.where(tf.less(predict_label, 0.5), 0, 1))
                  / predict_label.shape[0])
            print("第{}个epoch第{}个文件的roc_score为:{} 准确率为:{}".format(epoch, count,
                                                                               metrics.roc_auc_score(tf.nn.relu(
                                                                                   validation_label), predict_label),
                                                                               metrics.accuracy_score(
                                                                                   tf.nn.relu(validation_label),
                                                                                   tf.where(tf.less(predict_label, 0.5),
                                                                                            0, 1))))

        # 检查是否需要保存模型
        roc_val = metrics.roc_auc_score(tf.nn.relu(validation_label), predict_label)
        if roc < roc_val:
            roc = roc_val
            if epoch > 3:
                # 保存左端点神经网络模型
                model_user.save("./modeluser", save_format="tf")
                cmd = "s3cmd put -r ./modeluser " + args.model_output
                os.system(cmd)
                print("model_user已保存!")

                # 保存右端点神经网络模型
                model_club.save("./modelclub", save_format="tf")
                cmd = "s3cmd put -r ./modelclub " + args.model_output
                os.system(cmd)
                print("model_club已保存!")
        #更新loss结果
        if loss < before_loss:
            before_loss = loss
            stop_num = 0
        else:
            stop_num = stop_num + 1
            if stop_num >= args.stop_num:
                print("Early stop!")
                break


def write(data, count, args):
    # data为一个list类型
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

#读取预测数据
def read_test_data(count, file):
    print("开始读取第{}个文件数据! {}".format(count, datetime.datetime.now()))
    data = pd.read_csv("s3://" + file, sep=',', header=None).astype("str")
    #预测环境读取数据[id, 特征] 返回: [id, 数据data]
    id = data.iloc[:, 0]
    data = tf.convert_to_tensor(data.iloc[:, 1::].values, dtype=tf.float32)
    return id, data

def test_user(args):
    # 装载训练好的模型
    cmd = "s3cmd get -r  " + args.model_output + "modeluser"
    os.system(cmd)
    modeluser = keras.models.load_model("./modeluser", custom_objects={'tf': tf}, compile=False)
    print("Model_user is loaded!")
    # 读取数据
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    for file in input_files:
        # 读取数据 id为pandas.dataframe类型  data为tensor(tf.float32型)
        id, data = read_test_data(count, file)
        #第0列为玩家id, 第2列之后为输出特征
        result = np.zeros((data.shape[0], args.output_dim + 1)).astype("str")
        #写入id
        result[:, 0] = id.values
        predict = modeluser(data)
        result[:, 1:] = predict.numpy().astype("str")
        #写入结果
        write(result.tolist(), count, args)
        count = count + 1


def test_club(args):
    # 装载训练好的模型
    cmd = "s3cmd get -r  " + args.model_output + "modelclub"
    os.system(cmd)
    dnn_model = keras.models.load_model("./modelclub", custom_objects={'tf': tf}, compile=False)
    print("Model_club is loaded!")
    # 读取数据
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    for file in input_files:
        # 读取数据 id为pandas.dataframe类型  data为tensor(tf.float32型)
        id, data = read_test_data(count, file)
        #第0列为玩家id, 第2列为score
        result = np.zeros((data.shape[0], args.output_dim + 1)).astype("str")
        #写入id
        result[:, 0] = id.values
        predict = dnn_model(data)
        result[:, 1:] = predict.numpy().astype("str")
        #写入结果
        write(result.tolist(), count, args)
        count = count + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or test)", type=str, default='train_incremental')
    parser.add_argument("--split_dim", help="特征维数分界点(玩家特征: 1~split_dim, 战队特征: split_dim:)",
                        type=int, default=158)
    parser.add_argument("--lr", help="学习率", type=float, default=0.00001)
    parser.add_argument("--pos_weight", help="正样本的loss权重", type=float, default=30.0)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=30)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=200)
    parser.add_argument("--batch_size", help="batch的大小", type=int, default=3000)
    parser.add_argument("--input_dim", help="输入特征的维度", type=int, default=200)
    parser.add_argument("--feat_dim", help="隐含层神经元的数目", type=int, default=150)
    parser.add_argument("--feat_dim2", help="隐含层神经元的数目", type=int, default=100)
    parser.add_argument("--output_dim", help="输出特征的维度", type=int, default=64)
    parser.add_argument("--file_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=150000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='s3://models/gclmec/drecallcl/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')

    args = parser.parse_args()
    if args.env == "train" or args.env == "train_incremental":
        print("执行训练函数！")
        train(args)
    elif args.env == "test_user":
        print("注意:玩家只能有一列id!")
        print("对玩家进行特征变换输出!")
        test_user(args)
    else:
        print("注意:战队只能有一列id!")
        print("对俱乐部进行特征变换输出!")
        test_club(args)






