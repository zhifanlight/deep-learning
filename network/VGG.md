# $\mathrm{VGG}$

## 网络结构

### 堆叠卷积核

- $n$ 次 $\mathrm{stride} = 1$ 的 $k \times k$ 卷积相当于 $1$ 次 $\mathrm{stride} = 1$ 的 $t \times t$ 卷积

  $$
  t = n \cdot \left( k - 1 \right) + 1
  $$

- 一次 $5 \times 5$ 卷积相当于两次 $3 \times 3$ 卷积，一次 $7 \times 7$ 卷积相当于三次 $3 \times 3$ 卷积

- 当特征图维度相同时，两次 $3 \times 3$ 卷积的计算量小于 $5 \times 5$ 卷积的计算量，三次 $3 \times 3$ 卷积的计算量小于 $7 \times 7$ 卷积的计算量

  <center>
  <img src="images/stack_conv.png"/>
  </center>

### $\mathrm{VGG-16}$

- 共 $16$ 层，包括 $5$ 组卷积层，$3$ 个全连接层

  <center>
  <img src="images/vgg.png"/>
  </center>

#### 输入层 $\mathrm{Input}$

- 输入为 $224 \times 224$ 的 $\mathrm{RGB}$ 图像

#### 卷积层 $\mathrm{Conv1}$

- 对 $\mathrm{Input}$ 数据进行两次卷积，特征图维度为 $224 \times 224$

  - 两次卷积核结构相同：大小为 $3$，步长为 $1$，$\mathrm{padding}$ 值为 $1$

- 对特征图进行 $\mathrm{Pooling}$，特征图维度为 $112 \times 112$

  - 步长为 $2$ 的 $2 \times 2 \ \mathrm{max \ pooling}$

#### 卷积层 $\mathrm{Conv2}$

- 对 $\mathrm{Conv1}$ 结果进行两次卷积，特征图维度为 $112 \times 112$

  - 两次卷积核结构相同：大小为 $3$，步长为 $1$，$\mathrm{padding}$ 值为 $1$

- 对特征图进行 $\mathrm{Pooling}$，特征图维度为 $56 \times 56$

  - 步长为 $2$ 的 $2 \times 2 \ \mathrm{max \ pooling}$

#### 卷积层 $\mathrm{Conv3}$

- 对 $\mathrm{Conv2}$ 结果进行三次卷积，特征图维度为 $56 \times 56$

  - 三次卷积核结构相同：大小为 $3$，步长为 $1$，$\mathrm{padding}$ 值为 $1$

- 对特征图进行 $\mathrm{Pooling}$，特征图维度为 $28 \times 28$

  - 步长为 $2$ 的 $2 \times 2 \ \mathrm{max \ pooling}$

#### 卷积层 $\mathrm{Conv4}$

- 对 $\mathrm{Conv3}$ 结果进行三次卷积，特征图维度为 $28 \times 28$

  - 三次卷积核结构相同：大小为 $3$，步长为 $1$，$\mathrm{padding}$ 值为 $1$

- 对特征图进行 $\mathrm{Pooling}$，特征图维度为 $14 \times 14$

  - 步长为 $2$ 的 $2 \times 2 \ \mathrm{max \ pooling}$

#### 卷积层 $\mathrm{Conv5}$

- 对 $\mathrm{Conv4}$ 结果进行三次卷积，特征图维度为 $14 \times 14$

  - 三次卷积核结构相同：大小为 $3$，步长为 $1$，$\mathrm{padding}$ 为 $1$

- 对特征图进行 $\mathrm{Pooling}$，特征图维度为 $7 \times 7$

  - 步长为 $2$ 的 $2 \times 2 \ \mathrm{max \ pooling}$

#### 全连接层 $\mathrm{FC6}$

- 与 $\mathrm{Conv5}$ 进行全连接，生成向量维度为 $4096$

- 对生成向量先 $\mathrm{ReLU}$ 再 $\mathrm{Dropout}$，不改变生成向量维度

#### 全连接层 $\mathrm{FC7}$

- 与 $\mathrm{FC6}$ 进行全连接，生成向量维度为 $4096$

- 对生成向量先 $\mathrm{ReLU}$ 再 $\mathrm{Dropout}$，不改变生成向量维度

#### 全连接层 $\mathrm{FC8}$

- 与 $\mathrm{FC7}$ 进行全连接，生成向量维度为 $1000$

#### 输出层 $\mathrm{Output}$

- 对 $\mathrm{FC8}$ 生成向量进行 $\mathrm{softmax}$，计算每一类的概率

### $\mathrm{VGG-19}$

- 共 $19$ 层，包括 $5$ 组卷积层，$3$ 个全连接层

- 与 $\mathrm{VGG-16}$ 结构基本相同，区别在于 $\mathrm{Conv3}$、$\mathrm{Conv4}$ 与 $\mathrm{Conv5}$ 阶段均进行四次卷积

## 主要改进

- 将卷积核全部替换为 $3 \times 3$ 的小卷积核

  - 既减少了计算量，又增加了非线性拟合能力