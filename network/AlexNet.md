# $\mathrm{AlexNet}$

## 网络结构

- $5$ 个卷积层，$3$ 个全连接层

  <center>
  <img src="images/alexnet.png"/>
  </center>

- 原始 $\mathrm{AlexNet}$ 对特征图进行了 $\mathrm{GPU}$ 分组加速

  - 卷积层 $\mathrm{Conv1}$ 对输入图像进行卷积，$\mathrm{Conv3}$ 对 $\mathrm{Conv2}$ 的所有特征图进行卷积

  - 卷积层 $\mathrm{Conv2}$、$\mathrm{Conv4}$、$\mathrm{Conv5}$ 只对当前 $\mathrm{GPU}$ 的上一层特征图进行卷积

### 输入层 $\mathrm{Input}$

- 输入为 $227 \times 227$ 的 $\mathrm{RGB}$ 图像

### 卷积层 $\mathrm{Conv1}$

- 对 $\mathrm{Input}$ 数据进行卷积，特征图维度为 $55 \times 55$

  - 卷积核大小为 $11$，步长为 $4$，不进行 $\mathrm{padding}$

- 对特征图先 $\mathrm{ReLU}$ 再 $\mathrm{Pooling}$，特征图维度为 $27 \times 27$

  - 步长为 $2$ 的 $3 \times 3$ $\mathrm{max \ pooling}$

- 对特征图进行局部响应归一化，不改变特征图维度

### 卷积层 $\mathrm{Conv2}$

- 对 $\mathrm{Conv1}$ 结果进行卷积，特征图维度为 $27 \times 27$

  - 卷积核大小为 $5$，步长为 $1$，$\mathrm{padding}$ 为 $2$

- 对特征图先 $\mathrm{ReLU}$ 再 $\mathrm{Pooling}$，特征图维度为 $13 \times 13$

  - 步长为 $2$ 的 $3 \times 3$ $\mathrm{max pooling}$

- 对特征图进行局部响应归一化，不改变特征图维度

### 卷积层 $\mathrm{Conv3}$

- 对 $\mathrm{Conv2}$ 结果进行卷积，特征图维度为 $13 \times 13$

  - 卷积核大小为 $3$，步长为 $1$，$\mathrm{padding}$ 为 $1$

- 对特征图进行 $\mathrm{ReLU}$，不改变特征图维度

### 卷积层 $\mathrm{Conv4}$

- 对 $\mathrm{Conv3}$ 结果进行卷积，特征图维度为 $13 \times 13$

  - 卷积核大小为 $3$，步长为 $1$，$\mathrm{padding}$ 为 $1$

- 对特征图进行 $\mathrm{ReLU}$，不改变特征图维度

### 卷积层 $\mathrm{Conv5}$

- 对 $\mathrm{Conv4}$ 结果进行卷积，特征图维度为 $13 \times 13$

  - 卷积核大小为 $3$，步长为 $1$，$\mathrm{padding}$ 为 $1$

- 对特征图先 $\mathrm{ReLU}$ 再 $\mathrm{Pooling}$，特征图维度为 $6 \times 6$

  - 步长为 $2$ 的 $3 \times 3$ $\mathrm{max \ pooling}$

### 全连接层 $\mathrm{FC6}$

- 与 $\mathrm{Conv5}$ 进行全连接，生成向量维度为 $4096$

- 对生成向量先 $\mathrm{ReLU}$ 再 $\mathrm{Dropout}$，不改变生成向量维度

### 全连接层 $\mathrm{FC7}$

- 与 $\mathrm{FC6}$ 进行全连接，生成向量维度为 $4096$

- 对生成向量先 $\mathrm{ReLU}$ 再 $\mathrm{Dropout}$，不改变生成向量维度

### 全连接层 $\mathrm{FC8}$

- 与 $\mathrm{FC7}$ 进行全连接，生成向量维度为 $1000$

### 输出层 $\mathrm{Output}$

- 对 $\mathrm{FC8}$ 生成向量进行 $\mathrm{softmax}$，计算每一类的概率