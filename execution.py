#!/usr/bin/python3.6
# -*- coding: utf-8 -*-
# @Time    : 2022/12/26 20:32
# @Author  : Liangliang
# @File    : execution.py
# @Software: PyCharm

import gclmec
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='算法的参数')
    parser.add_argument("--env", help="运行的环境(train or test)", type=str, default='train_incremental')
    parser.add_argument("--flags", help="指示BatchNormalization的training参数(True or False)",
                        type=bool, default=True)
    parser.add_argument("--lr", help="学习率", type=float, default=0.0001)
    parser.add_argument("--stop_num", help="执行Early Stopping的最低epoch", type=int, default=10)
    parser.add_argument("--epoch", help="迭代次数", type=int, default=200)
    parser.add_argument("--sample_num", help = "采样子图的子图数目", type = int, default = 1000)
    parser.add_argument("--batch_num", help="打印loss函数的周期", type=int, default=1000)
    parser.add_argument("--input_dim", help="输入特征的维度", type=int, default=14)
    parser.add_argument("--feat_dim", help="隐含层神经元的数目", type=int, default=100)
    parser.add_argument("--output_dim", help="输出特征的维度", type=int, default=64)
    parser.add_argument("--subgraph_nodes_max_num", help="采样子图最大的节点数目", type=int, default=20000)
    parser.add_argument("--subgraph_edges_max_num", help="采样子图最大的边数目", type=int, default=200000)
    parser.add_argument("--subgraph_nodes_min_num", help="采样子图最小的节点数目", type=int, default=100)
    parser.add_argument("--subgraph_edges_min_num", help="采样子图最小的边数目", type=int, default=120)
    parser.add_argument("--k_hop", help="采样子图的跳连数目", type=int, default=1)
    parser.add_argument("--tau", help="动量更新的权值", type=int, default=0.95)
    parser.add_argument("--nodes_num", help="采样子图的节点数目", type=int, default=1)
    parser.add_argument("--file_nodes_max_num", help="单个csv文件中写入数据的最大行数", type=int, default=150000)
    parser.add_argument("--data_input", help="输入数据的位置", type=str, default='')
    parser.add_argument("--data_output", help="数据的输出位置", type=str, default='')
    parser.add_argument("--model_output", help="模型的输出位置",
                        type=str, default='s3://models/gclmec/')
    parser.add_argument("--tb_log_dir", help="日志位置", type=str, default='')
    args = parser.parse_args()
    if args.env == "train" or args.env == "train_incremental":
        gclmec.train(args)
    elif args.env == "test":
        gclmec.test(args)
    else:
        print("输入的环境参数错误, env只能为train或test!")