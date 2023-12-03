#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/12/26 20:33
# @Author  : hmliangliang
# @File    : gclmec.py
# @Software: PyCharm

import os
import datetime
import time
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras as keras
import s3fs

os.system("pip install dgl dglgo -f https://data.dgl.ai/wheels/repo.html")
os.environ['DGLBACKEND'] = "tensorflow"

os.environ['TF_FORCE_GPU_ALLOW_GROWTH'] = "true"
import dgl
from dgl import nn as nn

# 设置随机种子点
#random.seed(921208)
np.random.seed(921208)
tf.random.set_seed(921208)
os.environ['PYTHONHASHSEED'] = "921208"
# 设置GPU随机种子点
os.environ['TF_DETERMINISTIC_OPS'] = '1'


#读取文件系统
class S3FileSystemPatched(s3fs.S3FileSystem):
    def __init__(self, *k, **kw):
        super(S3FileSystemPatched, self).__init__(*k,
                                                  key=os.environ['AWS_ACCESS_KEY_ID'],
                                                  secret=os.environ['AWS_SECRET_ACCESS_KEY'],
                                                  client_kwargs={'endpoint_url': 'http://' + os.environ['S3_ENDPOINT']},
                                                  **kw
                                                  )


#定义resnet结构
class BlockNet(keras.Model):
    def __init__(self, inputs_dim):
        super(BlockNet, self).__init__()
        self.cov1 = keras.layers.Dense(inputs_dim, use_bias=True)
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.cov2 = keras.layers.Dense(inputs_dim*2, use_bias=True)
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        self.cov3 = keras.layers.Dense(inputs_dim, use_bias=True)

    def call(self, inputs, training=None, mask=None):
        h = self.cov1(inputs)
        h = self.batch_normalization1(h, training=training)
        h = tf.nn.relu(h)
        h = self.cov2(h)
        h = self.batch_normalization2(h, training=training)
        h = tf.nn.relu(h)
        h = self.cov3(h)
        h = h + inputs
        h = tf.nn.relu(h)
        return h


class ResNet(keras.Model):
    def __init__(self, output_dim):
        super(ResNet, self).__init__()
        self.res_cov1 = keras.layers.Dense(1024, activation = "relu", use_bias=True)
        self.res_dropout1 = keras.layers.Dropout(0.15)
        self.res_layers1 = BlockNet(1024)

        self.res_cov2 = keras.layers.Dense(800, activation = "relu", use_bias=True)
        self.res_dropout2 = keras.layers.Dropout(0.15)
        self.res_layers2 = BlockNet(800)

        self.res_cov3 = keras.layers.Dense(600, activation = "relu", use_bias=True)
        self.res_dropout3 = keras.layers.Dropout(0.15)
        self.res_layers3 = BlockNet(600)

        self.res_cov4 = keras.layers.Dense(512, activation = "relu", use_bias=True)
        self.res_dropout4 = keras.layers.Dropout(0.15)
        self.res_layers4 = BlockNet(512)

        self.res_cov5 = keras.layers.Dense(output_dim, activation = "relu", use_bias=True)
        self.res_dropout5 = keras.layers.Dropout(0.15)
        self.res_layers5 = BlockNet(output_dim)

    def call(self, inputs, training=None, mask=None):
        h = self.res_cov1(inputs)
        if training:
            h = self.res_dropout1(h, training=training)
        h = self.res_layers1(h)

        h = self.res_cov2(h)
        if training:
            h = self.res_dropout2(h, training=training)
        h = self.res_layers2(h)

        h = self.res_cov3(h)
        if training:
            h = self.res_dropout3(h, training=training)
        h = self.res_layers3(h)

        h = self.res_cov4(h)
        if training:
            h = self.res_dropout4(h, training=training)
        h = self.res_layers4(h)

        h = self.res_cov5(h)
        if training:
            h = self.res_dropout5(h, training=training)
        h = self.res_layers5(h)
        return h


#数据增广
def spectrum_feature_augmentation(H):
    '''SFA: Spectrum Feature Augmentation in Graph Contrastive Leanring and Beyond AAAI 2023'''
    k = 4
    d = H.shape[1]
    H_temp = tf.matmul(tf.transpose(H), H)
    H_temp = tf.nn.l2_normalize(H_temp, axis=1)
    r = tf.random.uniform((d, 1))
    for i in range(1, k + 1):
        r = tf.matmul(H_temp, r) / (k * d)
    r = tf.nn.l2_normalize(r, axis=1)
    H = H - tf.matmul(H, tf.matmul(r, tf.transpose(r))/d) / (tf.norm(r)**2 + 1e-6)
    H = tf.nn.l2_normalize(H, axis=1)
    return H


#定义teacher模型结构
class Teacher_Net(keras.Model):
    def __init__(self, input_dim = 300, feat_dim = 200, output_dim = 64):
        super(Teacher_Net, self).__init__()
        self.t_cov1 = nn.SAGEConv(input_dim, feat_dim, "mean")
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.t_cov2 = nn.SAGEConv(feat_dim, output_dim, "mean")
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()
        # self.t_layer = keras.layers.Dense(output_dim, activation = "sigmoid", use_bias=True)
        self.t_layer = ResNet(output_dim)

    def call(self, g, inputs, training = None, mask = None):
        inputs = self.t_cov1(g, inputs)
        inputs = self.batch_normalization1(inputs, training = training)
        inputs = tf.nn.leaky_relu(inputs)
        if training:
            inputs = spectrum_feature_augmentation(inputs)
        inputs = self.t_cov2(g, inputs)
        inputs = self.batch_normalization2(inputs, training = training)
        inputs = tf.nn.leaky_relu(inputs)
        if training:
            inputs = spectrum_feature_augmentation(inputs)
        inputs = self.t_layer(inputs, training = training)
        return inputs


#定义student模型结构
class Student_Net(keras.Model):
    def __init__(self, input_dim = 300, feat_dim = 200, output_dim = 64):
        super(Student_Net, self).__init__()
        self.t_cov1 = nn.SAGEConv(input_dim, feat_dim, "mean")
        self.batch_normalization1 = tf.keras.layers.BatchNormalization()
        self.t_cov2 = nn.SAGEConv(feat_dim, output_dim, "mean")
        self.batch_normalization2 = tf.keras.layers.BatchNormalization()

    def call(self, g, inputs, training = None, mask = None):
        inputs = self.t_cov1(g, inputs)
        inputs = self.batch_normalization1(inputs, training = training)
        inputs = tf.nn.leaky_relu(inputs)
        if training:
            inputs = spectrum_feature_augmentation(inputs)
        inputs = self.t_cov2(g, inputs)
        inputs = self.batch_normalization2(inputs, training = training)
        inputs = tf.nn.leaky_relu(inputs)
        if training:
            inputs = spectrum_feature_augmentation(inputs)
        return inputs

#读取数据
def read_graph_data(args):
    '''读取图数据部分
    二维数组，每一个元素为边id，一行构成一条边
    '''
    path = args.data_input.split(',')[0]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    count = 0
    print("开始读取数据! {}".format(datetime.datetime.now()))
    data = pd.DataFrame()
    for file in input_files:
        count = count + 1
        print("当前正在处理第{}个文件,文件路径:{}......".format(count, "s3://" + file))
        # 读取边结构数据
        data = pd.concat([data, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)

    # 读取属性特征信息
    # 最后一列为节点的类型，所以特征的列数为n-1
    path = args.data_input.split(',')[1]
    s3fs.S3FileSystem = S3FileSystemPatched
    fs = s3fs.S3FileSystem()
    input_files = sorted([file for file in fs.ls(path) if file.find("part-") != -1])
    data_attr = pd.DataFrame()
    for file in input_files:
        # 读取属性特征数据
        data_attr = pd.concat([data_attr, pd.read_csv("s3://" + file, sep=',', header=None)], axis=0)
    # 读取节点的属性特征数据
    data_attr = tf.convert_to_tensor(data_attr.values, dtype=tf.float32)
    #定义图结构
    g = dgl.graph((data.iloc[:, 0].to_list(), data.iloc[:, 1].to_list()), num_nodes=data_attr.shape[0],
                  idtype=tf.int32)
    #转化为无向图
    g = dgl.to_bidirected(g)
    g.ndata["feat"] = data_attr[:,:-1]
    g.ndata["type"] = tf.reshape(data_attr[:, -1],(-1, 1))
    g = dgl.add_self_loop(g)
    return g


# Maximum Entropy Coding Loss
def loss_function(Z_s, Z_t):
    '''计算Maximum Entropy Coding Loss
        参考文献: Liu X, Wang Z, Li Y L, et al. Self-Supervised Learning via Maximum Entropy Coding[C]
        //Advances in Neural Information Processing Systems (NeurIPS 2022).'''

    m = Z_s.shape[0]
    d = Z_s.shape[1]
    e = 0.06
    lamda = 1 / (m * e)
    v = (m + d) / 2
    k = 4
    loss = 0
    #对embedding进行l2_normalize
    Z_s = tf.linalg.l2_normalize(Z_s, axis=1)
    Z_t = tf.linalg.l2_normalize(Z_t, axis=1)

    Z = lamda * tf.matmul(tf.transpose(Z_s), Z_t)
    Z = tf.linalg.l2_normalize(Z, axis=1)
    pow_matrix = Z
    for i in range(1,k+1):
        if i > 1:
            pow_matrix = tf.matmul(pow_matrix, Z)
            pow_matrix = tf.linalg.l2_normalize(pow_matrix, axis=1)
        loss = loss - v * (math.pow(-1, i + 1)/i) * tf.linalg.trace(pow_matrix)
    loss = loss / (m*d)
    return loss

#定义训练模型方法
def train(args):
    #读取图数据
    g = read_graph_data(args)
    #训练，定义模型结构
    if args.env == "train":
        # teacher网络
        model_t = Teacher_Net(args.input_dim, args.feat_dim, args.output_dim)
        #student网络
        model_s = Student_Net(args.input_dim, args.feat_dim, args.output_dim)
    else:
        # 装载训练好的模型
        '''装载数据模型'''
        model_t = Teacher_Net(args.input_dim, args.feat_dim, args.output_dim)
        model_s = Student_Net(args.input_dim, args.feat_dim, args.output_dim)
        #装载teacher net模型
        cmd = "s3cmd get -r  " + args.model_output + "Teacher_Net"
        os.system(cmd)
        checkpoint_path = "./Teacher_Net/Teacher_Net.pb"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model_t.load_weights(latest)
        print("Teacher Model weights load!")

        # 装载student net模型
        cmd = "s3cmd get -r  " + args.model_output + "Student_Net"
        os.system(cmd)
        checkpoint_path = "./Student_Net/Student_Net.pb"
        checkpoint_dir = os.path.dirname(checkpoint_path)
        latest = tf.train.latest_checkpoint(checkpoint_dir)
        model_s.load_weights(latest)
        print("Student Model weights load!")

    #定义优化器
    optimizer = keras.optimizers.Adam(learning_rate=args.lr)
    BEFORE_LOSS = 2 ** 32
    STOP_NUM = 0
    loss = 0
    for epoch in range(args.epoch):
        for sample_nums in range(args.sample_num):
            k_hop = args.k_hop
            if sample_nums % args.batch_num == 0:
                print("开始第{}个epoch第{}个子图的训练 {}".format(epoch, sample_nums, datetime.datetime.now()))
            # 对大规模图进行采样
            g_sub, _ = dgl.khop_in_subgraph(g, np.random.randint(0, g.ndata["feat"].shape[0], args.nodes_num).tolist(),
                                            k=args.k_hop)
            #防止采样的子图规模过大或过小
            while (g_sub.num_nodes() >= args.subgraph_nodes_max_num or g_sub.num_edges()
                   >= args.subgraph_edges_max_num) and k_hop >=1 and k_hop <= 3 and (g_sub.num_nodes()
                                                                      < args.subgraph_nodes_min_num or
                                                                      g_sub.num_edges() < args.subgraph_edges_min_num):
                if g_sub.num_nodes() >= args.subgraph_nodes_max_num or g_sub.num_edges() >= args.subgraph_edges_max_num:
                    print("采用的子图规模过大，可能会内存溢出，采样子图的节点数目为:{} 子图边的数目:{} {} ".format(epoch,
                                                                                 sample_nums, g_sub.num_nodes(),
                                                                                 g_sub.num_edges(),
                                                                                 datetime.datetime.now()))
                    #减小阶数
                    k_hop = k_hop - 1
                elif g_sub.num_nodes() < args.subgraph_nodes_min_num or g_sub.num_edges() < args.subgraph_edges_min_num:
                    print("采用的子图规模过小，采样子图的节点数目为:{} 子图边的数目:{} {} ".format(epoch, sample_nums,
                                                                         g_sub.num_nodes(), g_sub.num_edges(),
                                                                         datetime.datetime.now()))
                    # 增大阶数采样子图
                    k_hop = k_hop - 1
                if k_hop >= 1:
                    g_sub, _ = dgl.khop_in_subgraph(g, np.random.randint(0, g.ndata["feat"].shape[0],
                                                                         args.nodes_num).tolist(), k=k_hop)
                    if sample_nums % args.batch_num == 0:
                        print("第{}个epoch 第{}个子图采用完成 子图的节点数目为:{} 子图边的数目:{} {} ".format(epoch,
                                                                                      sample_nums,
                                                                              g_sub.num_nodes(), g_sub.num_edges(),
                                                                              datetime.datetime.now()))
                #k_hop <= 0终止采样过程
                else:
                    break
            if g_sub.num_nodes() <= args.subgraph_nodes_max_num and g_sub.num_edges() <= args.subgraph_edges_max_num:
                with tf.GradientTape() as tape:
                    Z_s = model_t(g_sub, g_sub.ndata["feat"], training=True)
                    Z_t = model_s(g_sub, g_sub.ndata["feat"], training=True)
                    loss = loss_function(Z_s, Z_t)
                gradients = tape.gradient(loss, model_t.trainable_variables)
                optimizer.apply_gradients(zip(gradients, model_t.trainable_variables))
                if sample_nums % args.batch_num == 0:
                    print("第{}个epoch第{}个子图的训练梯度更新完成 {}".format(epoch, sample_nums, datetime.datetime.now()))

                # 动量更新student网络参数 参考https://github.com/garder14/byol-tensorflow2/blob/main/pretraining.py
                #更新student中t_cov1网络的参数
                model_s_weights = model_s.t_cov1.get_weights()
                for layer in range(len(model_s_weights)):
                    model_s_weights[layer] = args.tau * model_s_weights[layer] + (1 - args.tau) * \
                                                 model_t.t_cov1.get_weights()[layer]
                model_s.t_cov1.set_weights(model_s_weights)

                # 更新student中t_cov2网络的参数
                model_s_weights2 = model_s.t_cov2.get_weights()
                for layer in range(len(model_s_weights2)):
                    model_s_weights2[layer] = args.tau * model_s_weights2[layer] + (1 - args.tau) * \
                                             model_t.t_cov2.get_weights()[layer]
                model_s.t_cov2.set_weights(model_s_weights2)

                #更新student网络中batch_normalization1的参数
                net_student_batch_weights = model_s.batch_normalization1.get_weights()
                for layer in range(len(net_student_batch_weights)):
                    net_student_batch_weights[layer] = args.tau * net_student_batch_weights[layer] + \
                                                       (1 - args.tau) * \
                                                       model_t.batch_normalization1.get_weights()[layer]
                model_s.batch_normalization1.set_weights(net_student_batch_weights)

                # 更新student网络中bach_normalization2的参数
                net_student_batch_weights2 = model_s.batch_normalization2.get_weights()
                for layer in range(len(net_student_batch_weights2)):
                    net_student_batch_weights2[layer] = args.tau * net_student_batch_weights2[layer] + (1 - args.tau) * \
                                                       model_t.batch_normalization2.get_weights()[layer]
                model_s.batch_normalization2.set_weights(net_student_batch_weights2)
                if sample_nums % args.batch_num == 0:
                    print("第{}个epoch第{}个子图的神经网络训练梯度更新完成 loss:{} {}".format(epoch,
                                                                           sample_nums, loss,
                                                                           datetime.datetime.now()))

        if loss < BEFORE_LOSS:
            BEFORE_LOSS = loss
            STOP_NUM = 0
        else:
            STOP_NUM = STOP_NUM + 1
            if STOP_NUM >= args.stop_num:
                print("Stop early!")
                break
    # 保存图神经网络模型
    # 保存teacher net
    # model_t.summary()
    model_t.save_weights("./Teacher_Net/Teacher_Net.pb", save_format="tf")
    print("teacher net已保存!")
    cmd = "s3cmd put -r ./Teacher_Net " + args.model_output
    os.system(cmd)

    # 保存student net
    # net.summary()
    model_s.save_weights("./Student_Net/Student_Net.pb", save_format="tf")
    print("student net已保存!")
    cmd = "s3cmd put -r ./Student_Net " + args.model_output
    os.system(cmd)
    flag = True
    return flag
    print("模型保存完毕! {}".format(datetime.datetime.now()))


def write(data, args, count):
    #注意在此业务中data是一个二维list
    start = time.time()
    #数据的数量
    n = len(data)
    if n > 0:
        with open(os.path.join(args.data_output, 'pred_{}.csv'.format(count)), mode="a") as resultfile:
            # 说明此时的data是[[],[],...]的二级list形式
            for i in range(n):
                line = ",".join(map(str, data[i])) + "\n"
                resultfile.write(line)
    cost = time.time() - start
    print("第{}个大数据文件已经写入完成,写入数据的行数{} 耗时:{}  {}".format(count, n, cost, datetime.datetime.now()))


#计算节点的embedding
def get_embedding(g, model, args, nodes_list, N, count):
    result = np.zeros((len(nodes_list), args.output_dim + 2)).astype("str") #id + label + args.output_dim
    num = 0
    for i in nodes_list:
        num = num + 1
        # 对大规模图进行采样
        if num % 100 == 0:
            print("一共有{}个节点，当前正在处理第{}个节点 {}".format(N, count * args.file_nodes_max_num + num,
                                                    datetime.datetime.now()))
        g_sub, _ = dgl.khop_in_subgraph(g, i, k=args.k_hop)
        k = args.k_hop
        flag = True
        while (g_sub.num_nodes() >= args.subgraph_nodes_max_num or g_sub.num_edges() >= args.subgraph_edges_max_num) \
                and k >= 1:
            if g_sub.num_nodes() >= args.subgraph_nodes_max_num or g_sub.num_edges() >= args.subgraph_edges_max_num:
                #减少阶数防止图规模过大
                k = k - 1
                print("i={} k={}采样的子图过大,可能会造成内存溢出，正在重新采样子图!{}".format(i, k, datetime.datetime.now()))
            #当度为1的子图仍然超过规模，则不计算当前节点的嵌入
            if k < 1:
                flag = False
                break
            else:
                g_sub, _ = dgl.khop_in_subgraph(g, i, k=k)
                print("i={} k={}采样的子图节点数为:{}  边数为:{}  {}".format(i, k, g_sub.num_nodes(),
                                                                    g_sub.num_edges(), datetime.datetime.now()))
        if flag:
            embed = model(g_sub, g_sub.ndata["feat"])
            # 获取中心节点在原始图中真实的编号与当前子图中编号间的映射关系
            #g_sub.ndata[dgl.NID] --> <tf.Tensor: shape=(3,), dtype=int32, numpy=array([0, 1, 4])> 第j个值即为原始图真实ID
            #j为抽样子图中节点的编号
            j = tf.where(g_sub.ndata[dgl.NID] == i)[0, 0]
            label = g_sub.ndata["type"][j, 0].numpy()
            #输出的特征格式[id,节点的类型(数字标注), 特征向量...]
            result[num -1, 0] = str(i)
            result[num -1, 1] = str(label)
            result[num -1, 2::] = embed[j, :].numpy().astype("str")
            if num % 100 == 0:
                print("已完成第{}个节点的embedding vectors 时间:{}".format(i, datetime.datetime.now()))
        else:
            result[num - 1, 0] = str(i)
            label = g.ndata["type"][i, 0].numpy()
            result[num - 1, 1] = str(label)
            print("第{}个节点图采样跳过!  {}".format(i, datetime.datetime.now()))
    #把结果写入文件系统中
    write(result.tolist(), args, count)


#计算embedding
def test(args):
    #装载训练好的模型
    '''装载数据模型'''
    model = Student_Net(args.input_dim, args.feat_dim, args.output_dim)

    cmd = "s3cmd get -r  " + args.model_output + "Student_Net"
    os.system(cmd)
    checkpoint_path = "./Student_Net/Student_Net.pb"
    checkpoint_dir = os.path.dirname(checkpoint_path)
    latest = tf.train.latest_checkpoint(checkpoint_dir)
    model.load_weights(latest)
    print("Model weights have been loaded!")

    #装载图数据
    g = read_graph_data(args)
    print("图构建完成, 图的节点数为:{} 边数为:{} {}".format(g.num_nodes(), g.num_edges(), datetime.datetime.now()))
    N = g.num_nodes()
    nodes_list = []
    element_num = -1
    for element in range(N):
        element_num = element_num + 1
        nodes_list.append(element)
        if ((element_num + 1) % args.file_nodes_max_num == 0) or (element_num == N - 1):
            id = int((element_num + 1) / args.file_nodes_max_num)
            get_embedding(g, model, args, nodes_list, N, id)
            nodes_list = []
    print("所有节点的embedding已输出！ {}".format(datetime.datetime.now()))
