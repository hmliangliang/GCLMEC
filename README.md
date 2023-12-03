# GCLMEC
Graph contrastive learning with minimum entropy coding

文中的PDF为本文算法关键idea的参考文献

该算法给出了一整套部署线上的流程，包含了graph embedding(execution.py)、基于faiss的向量检索(drecall.py)、精排(dnnrank.py)三个不同时期的代码

featureselection.py为一个基于xgboost的特征选择工具，只要是对特征的重要性进行打分，筛选出重要的特征
