<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 数据集

## MNIST

- 手写数字数据集，每张图片都是 \\(28 \times 28\\) 的灰度图

- 每张图片对应 0～9 中的一个数字，有相应类别标签

- 训练集 60000 张，测试集 10000 张

- 主页链接：http://yann.lecun.com/exdb/mnist/

## CIFAR

### CIFAR-10

- 普通物体数据集，每张图片都是 \\(32 \times 32\\) 的彩色图

- 共 10 类，每张图片仅对应 1 类，有相应类别标签

- 训练集 50000 张，每类 5000 张；测试集 10000 张，每类 1000 张

- 主页链接：https://www.cs.toronto.edu/~kriz/cifar.html

### CIFAR-100

- 普通物体数据集，每张图片都是 \\(32 \times 32\\) 的彩色图

- 共 20 大类，每个大类细分为 5 个小类，共 100 小类

- 每张图片对应 1 类，既有小类标签，也有对应大类标签

- 训练集 50000 张，每小类 500 张；测试集 10000 张，每小类 100 张

- 主页链接：https://www.cs.toronto.edu/~kriz/cifar.html

## CelebA

- 人脸数据集，经过裁剪的每张图片都是 \\(178 \times 218\\) 的彩色图

- 共 202599 张，大致分布如下：

	- 训练集 160000 张

	- 验证集 20000 张

	- 测试集 20000 张

- 每张图片标记 40 个人脸属性，以及每个属性下的正、负样本类别

- 每张图片标记 5 个特征点坐标：左眼、右眼、鼻尖、左嘴角、右嘴角

- 主页链接：mmlab.ie.cuhk.edu.hk/projects/CelebA.html