<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 聚类

## 背景介绍

- 将无标签样本分成多个类或多个簇，使得同一簇内的样本具有高度相似性，不同簇间的样本具有较大差异性

- 属于无监督学习

## K-Means

### 基本思想

- K-Means 的目标是数据集中的 \\(n\\) 个样本分成 \\(k\\) 个簇，同时要满足：

	- 每个簇内至少一个样本

	- 每个样本必须属于一个簇，而且只能属于一个簇

### 算法流程

- 适当选择 \\(k\\) 个初始聚类中心

- 在每一次迭代中，计算每个样本到各个中心的距离，将其归到距离最近的类中

- 根据当前划分，更新每一个簇的中心值

- 重复上述两步，直到某次更新后所有类的中心不再改变

### 算法分析

- 当簇间区别明显，且每个簇大小相近时，聚类效果比较明显

- 需要事先指定 \\(k\\) 值，但很多情况并不知道应该划分为几个类最合适

- 需要人为确定初始聚类中心，不同的初始值可能会导致完全不同的聚类结果

## K-Means ++

### 基本思想

- 自动选择 \\(k\\) 个聚类中心，并使初始聚类中心间的距离要尽可能远

- 根据选出的 \\(k\\) 个聚类中心，执行普通的 K-Means 

### 算法流程

- 随机选择一个样本作为第一个聚类中心

- 对数据集中每个样本，计算其到最近聚类中心的距离 \\(D(x)\\)

- 选择一个新样本作为新的聚类中心，原则是：

	- \\(D(x)\\) 越大的点，被选为聚类中心的概率越大

- 重复上述两步，直到选出 \\(k\\) 个聚类中心

- 根据选出的 \\(k\\) 个聚类中心，执行普通的 K-Means 

#### 选择聚类中心

- 由于存在噪声，不能直接选择最大的 \\(D(x)\\) 对应的样本

- 为保证 \\(D(x)\\) 越大的点被选中概率越大，选择方式如下：

	- 计算所有 \\(D(x)\\) 的和 \\(S\\)

	- 随机生成一个 \\([0,S)\\) 范围内的数 \\(R\\)，将包含 \\(R\\) 的样本作为新的聚类中心

- 上述过程是把每个 \\(D(x)\\) 看成长度为 \\(D(x)\\) 的线段，拼接后随机选择一点，并选中该点所在线段；某条线段越长，其被选中的概率越大

### 算法分析

- 解决了 K-Means 手动选择初始聚类中心的问题，聚类效果较稳定

- 依然没有解决需要手动设置聚类数目 \\(k\\) 的问题

## kNN

- kNN 实际上是分类算法的一种，属于有监督学习，但名字上容易与 K-Means 混淆

### 基本思想

- 如果某样本的 \\(k\\) 个最相似样本中的大多数都属于某个类别，那么该样本也很有可能属于改类别

### 算法流程

- 在特征空间中，寻找当前样本的前 \\(k\\) 个最相似样本

- 将这 \\(k\\) 个样本中出现次数最多的类别标签作为当前样本的类别标签

### 算法分析

- 没有明显的前期训练过程，将样本进行简单特征提取后，即可进行分类