# 训练技巧

## 数据增强

- 参考 [$\mathrm{DataPreprocess.md}$](../basic/DataPreprocess.md)

## 权重初始化

### 输入为固定高斯分布 

- 对于输入 $x$，假设 $\mathbb{E} \left( x \right) = 0, \ \mathrm{Var} \left( x \right) = 1$

- 假设输入为 $n$ 维，输出为 $1$ 维，激活函数为 $\mathrm{Sigmoid}$，权重 $w \sim N \left( 0, \ 1 \right) $：

  $$
  y = \sum_{i = 1}^{n} w_{i} x_{i} \qquad z = \frac{1}{1 + \mathrm{e}^{-y}}
  $$

- 由期望、方差的线性运算可知：

  $$
  \mathbb{E} \left( y \right) = \sum_{i = 1}^{n} \mathbb{E} \left( x_{i} \right) \cdot \mathbb{E} \left( w_{i} \right) = 0
  $$

  $$
  \mathrm{Var} \left( y \right) = \sum_{i = 1}^{n} \mathrm{Var} \left( x_{i} \right) \cdot \mathrm{Var} \left( w_{i} \right) = n
  $$

  - 关于期望、方差线性运算，参考 [$\mathrm{ProbabilityBasis.md}$](../basic/ProbabilityBasis.md)

- 线性加权 $y$ 的方差与输入维度有关，很容易落入 $\mathrm{Sigmoid}$ 饱和区，不利于反向传播

### 输出为固定高斯分布

- 对于输入 $x$，假设 $\mathbb{E} \left( x \right) = 0, \ \mathrm{Var} \left( x \right) = 1$

- 假设输入为 $n$ 维，输出为 $1$ 维，激活函数为 $\mathrm{Sigmoid}$，权重 $w \sim N \left( 0, \ \sqrt{\frac{1}{n}} \right) $：

  $$
  y = \sum_{i = 1}^{n} w_{i} x_{i} \qquad z = \frac{1}{1 + \mathrm{e}^{-y}}
  $$

- 由期望、方差的线性运算可知：

  $$
  \mathbb{E} \left( y \right) = \sum_{i = 1}^{n} \mathbb{E} \left( x_{i} \right) \cdot \mathbb{E} \left( w_{i} \right) = 0
  $$

  $$
  \mathrm{Var} \left( y \right) = \sum_{i = 1}^{n} \mathrm{Var} \left( x_{i} \right) \cdot \mathrm{Var} \left( w_{i} \right) = 1
  $$

- 线性加权 $y$ 的方差与输入维度无关，落入 $\mathrm{Sigmoid}$ 饱和区概率较低，方便反向传播

### 输出为固定均匀分布

- 对于输入 $x$，假设 $\mathbb{E} \left( x \right) = 0, \ \mathrm{Var} \left( x \right) = 1$

- 假设输入为 $n$ 维，输出为 $1$ 维，激活函数为 $\mathrm{Sigmoid}$，权重 $w \sim U \left( -\sqrt{\frac{1}{n}}, \ \sqrt{\frac{1}{n}} \right) $

- 均匀分布的期望、方差计算如下：

  $$
  \mathbb{E} \left( x \right) = \frac{a + b}{2} \qquad \mathrm{Var} \left( x \right) = \frac{ \left( b - a \right)^{2}}{12}
  $$

- 由期望、方差的线性运算可知：

  $$
  \mathbb{E} \left( y \right) = \sum_{i = 1}^{n} \mathbb{E} \left( x_{i} \right) \cdot \mathbb{E} \left( w_{i} \right) = 0
  $$

  $$
  \mathrm{Var} \left( y \right) = \sum_{i = 1}^{n} \mathrm{Var} \left( x_{i} \right) \cdot \mathrm{Var} \left( w_{i} \right) = \frac{1}{3}
  $$

- 线性加权 $y$ 的方差与输入维度无关，落入 $\mathrm{Sigmoid}$ 饱和区概率较低，方便反向传播

### $\mathrm{Xavier}$

- 同时考虑输入、输出维度，保证输入和输出的方差尽量相等，使网络信息更好的流动

- 假设输入为 $n$ 维，输出为 $m$ 维，激活函数为 $\mathrm{ReLU}$ 或 $\mathrm{Tanh}$，权重 $w \sim U \left( -\sqrt{\frac{6}{m + n}}, \ \sqrt{\frac{6}{m + n}} \right)$

- 由期望、方差的线性运算可知：

  $$
  \mathbb{E} \left( y_{j} \right) = \sum_{i = 1}^{n} \mathbb{E} \left( x_{i} \right) \cdot \mathbb{E} \left( w_{i} \right) = 0
  $$

  $$
  \mathrm{Var} \left( y_{j} \right) = \sum_{i = 1}^{n} \mathrm{Var} \left( x_{i} \right) \cdot \mathrm{Var} \left( w_{i} \right) = \frac{2n}{m + n} \cdot \mathrm{Var} \left( x \right)
  $$

  - 当输入、输出维度近似相等时，$\mathrm{Var} \left( y \right) = \mathrm{Var} \left( x \right)$

- 激活函数是 $\mathrm{Tanh}$ 时，深层权重可以保持良好分布

- 激活函数是 $\mathrm{ReLU}$ 时，深层权重迅速向 $0$ 靠拢

### $\mathrm{MSRA}$

- 在 $\mathrm{ReLU}$ 网络中，假定每一层只有一半的神经元被激活

- 如果输入 $x$ 满足 $\mathbb{E} \left( x \right) = 0, \ \mathrm{Var} \left( x \right) = 1$，权重初始化为 $w \sim U \left( -\sqrt{\frac{2}{n}}, \ \sqrt{\frac{2}{n}} \right)$

  - 由期望、方差的线性运算可知：

    $$
    \mathbb{E} \left( y \right) = \frac{1}{2} \sum_{i = 1}^{n} \mathbb{E} \left( x_{i} \right) \cdot \mathrm{E} \left( w_{i} \right) = 0
    $$

    $$
    \mathrm{Var} \left( y \right) = \frac{1}{2} \sum_{i = 1}^{n} \mathrm{Var} \left( x_{i} \right) \cdot \mathrm{Var} \left( w_{i} \right) = \frac{1}{3}
    $$

- 如果输入 $x$ 不满足上述约束，权重初始化为 $w \sim U \left( -\sqrt{\frac{12}{m + n}}, \ \sqrt{\frac{12}{m + n}} \right)$

  - 由期望、方差的线性运算可知：

    $$
    \mathbb{E} \left( y_{j} \right) = \frac{1}{2} \sum_{i = 1}^{n} \mathbb{E} \left( x_{i} \right) \cdot \mathbb{E} \left( w_{i} \right) = 0
    $$

    $$
    \mathrm{Var} \left( y_{j} \right) = \frac{1}{2} \sum_{i = 1}^{n} \mathrm{Var} \left( x_{i} \right) \cdot \mathrm{Var} \left( w_{i} \right) = \frac{2n}{m + n} \cdot \mathrm{Var} \left( x \right)
    $$

## 正则化

- 通过 $L_{1}$ 或 $L_{2}$ 正则项，使参数大部分为 $0$ 或尽量接近 $0$，一定程度上简化模型

- 关于正则化，参考 [$\mathrm{Regularization.md}$](../basic/Regularization.md)

## $\mathrm{Dropout}$

- $\mathrm{Caffe}$ 实现：

  - 训练时以概率 $p$ 让某些神经元不工作，并将其他神经元输出扩大 $\frac{1}{1 - p}$ 倍

  - 测试时不进行 $\mathrm{Dropout}$

  - 需要设置 $\mathrm{dropout\_ratio}$，即神经元不工作的概率

- $\mathrm{Tensorflow}$ 实现：

  - 训练时以概率 $q$ 让某些神经元工作，并将其输出扩大为 $\frac{1}{q}$ 倍

  - 测试时不进行 $\mathrm{Dropout}$

  - 需要设置 $\mathrm{keep\_prob}$，即神经元工作的概率

- 使用 $\mathrm{Dropout}$ 层对预训练的网络进行 $\mathrm{fine \ tuning}$ 时，所有参数都要乘以 $\frac{1}{p}$

- 从正则化角度，$\mathrm{Dropout}$ 强迫一个神经元和随机挑选出来的神经元共同工作，可以减少神经元之间的依赖，增强泛化能力

- 从 $\mathrm{Bagging}$ 角度，每次 $\mathrm{Dropout}$ 得到的模型都不相同，不同模型之间的过拟合可能相互抵消，从而整体上减少过拟合

- 在使用 $\mathrm{Dropout}$ 后，反向传播只针对网络的一部分，参数更新比较随机，训练时间较长

## $\mathrm{LRN}$

- $\mathrm{Local \ Response \ Normalization}$，即局部响应归一化

### 侧抑制

- 在神经生物学中，被激活的神经元会抑制相邻神经元

- $\mathrm{LRN}$ 通过在同一层的不同特征通道之间进行归一化，实现侧抑制

### 计算过程

- $N$ 是当前层总的特征通道数

- $a_{x, \ y}^{i}$ 表示第 $i$ 个通道 $\left( x, \ y \right)$ 位置的激活输出，$b_{x, \ y}^{i}$ 表示 $a_{x, \ y}^{i}$ 的 $\mathrm{LRN}$ 输出

- $k = 2, \ n = 5, \ \alpha = 1e-4, \ \beta = 0.75$ 均为超参数

- $\mathrm{LRN}$ 计算如下：

  $$
  b_{x, \ y}^{i} = a_{x, \ y}^{i} \ / \ \left( k + \alpha \sum_{j = \max{\left( 0, \ i - n / 2 \right)}}^{\min{\left( N - 1, \ i + n / 2 \right)}} \left( a_{x, \ y}^{j} \right)^{2} \right) ^{\beta}
  $$

  - 因此 $b_{x, \ y}^{i}$ 是以 $i$ 通道为中心的 $n$ 个相邻通道上， $\left( x, \ y \right)$ 位置激活值的某种组合

  - $a_{x,y}^{i}$ 越大，相邻通道 $\left( x, \ y \right)$ 位置上的输出被抑制得越严重

### 工程实现

- 在 $\mathrm{Caffe}$ 中有两种归一化方式：通道内、通道间

- $\mathrm{LRN}$ 层通常放在 $\mathrm{ReLU}$ 层之后

## $\mathrm{Batch \ Normalization}$

- 通过固定每一层的输入分布，加速网络收敛

- 关于 $\mathrm{Batch \ Normalization}$，参考 [$\mathrm{BatchNormalization.md}$](BatchNormalization.md)