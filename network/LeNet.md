<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# LeNet

## 网络结构

- 经典的 LeNet 结构如下，从输入到输出一共 \\(8\\) 层

	![img](images/lenet.png)

- 但 Caffe 与 Tensorflow 中的实现方式略有不同

	- 输入图片为 \\(28 \times 28\\)

	- 每一个卷积层的卷积核个数不同

	- 卷积前进行 padding，卷积前后维度不变

	- 两个全连接层合并为一个更高维的全连接层

### 输入层 Input

- 输入为 \\(32 \times 32\\) 的灰度图

### 卷积层 Conv1

- 对 Input 数据进行卷积，生成的特征图维度为 \\(28 \times 28\\)

	- 卷积核为 \\(5\\)，步长为 \\(1\\)，不进行 padding

### 池化层 Pool1

- 对 Conv1 结果进行下采样，生成的特征图维度为 \\(14 \times 14\\)

### 卷积层 Conv2

- 对 Pool1 结果进行卷积，生成的特征图维度为 \\(10 \times 10\\)

	- 卷积核为 \\(5\\)，步长为 \\(1\\)，不进行 padding

### 池化层 Pool2

- 对 Conv2 结果进行下采样，生成的特征图维度为 \\(5 \times 5\\)

### 全连接层 FC1

- 与 Pool2 结果建立全连接，生成的向量维度为 \\(120\\) 维

### 全连接层 FC2

- 与 FC1 建立全连接，生成的向量维度为 \\(84\\) 维

### 输出层 Output

- 与 FC2 建立全连接，生成的向量维度为 \\(10\\) 维

	- 生成的向量经过 Softmax 处理，分别代表每一类的概率