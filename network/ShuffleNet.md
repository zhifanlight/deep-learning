# $\mathrm{ShuffleNet}$

## $\mathrm{ShuffleNet \ v1}$

- 实际运行速度比 $\mathrm{AlexNet}$ 快 $13$ 倍，精度与 $\mathrm{VGG}$ 持平

### 设计思想

- 通过分组 $\mathrm{pointwise}$ 卷积减少普通 $\mathrm{pointwise}$ 卷积的计算量

- 通过 $\mathrm{channel \ shuffle}$，加强不同组之间的信息交流，提升模型的表示能力

### $\mathrm{ShuffleNet}$ 单元

- 采用 $\mathrm{ResNet}$ 的 $\mathrm{bottleneck}$ 思想

  - 将输入通道分为 $g$ 组，每组 $n$ 个通道

  - 在每一组内进行 $\mathrm{pointwise}$ 卷积，减少特征图通道数

  - 通过 $\mathrm{channel \ shuffle}$ 加强组间信息交流：

    - 将特征图通道 $\mathrm{reshape}$ 成新维度 $\left( g, \ n \right)$

    - 对上述通道进行转置，得到 $\left( n, \ g \right)$

    - 将通道展开，得到 $n \times g$ 个特征图

    - 把相邻 $n$ 个通道划分为一组

  - 进行 $\mathrm{depthwise}$ 卷积

  - 在每一组内进行 $\mathrm{pointwise}$ 卷积

- 当 $\mathrm{depthwise}$ 卷积步长为 $1, \ 2$ 时，$\mathrm{shufflenet}$ 单元的结构分别如下：

  - 步长为 $1$ 时，卷积分支与 $\mathrm{shortcut}$ 分支直接相加，第二个 $\mathrm{pointwise}$ 卷积是为了匹配 $\mathrm{shortcut}$ 特征图的通道数

  - 步长为 $2$ 时，卷积分支与 $\mathrm{shortcut}$ 分支按通道拼接，$\mathrm{shortcut}$ 分支通过步长为 $2$、核为 $3$ 的 $\mathrm{average-pooling}$ 进行下采样

  - 去掉 $\mathrm{depthwise}$ 卷积和第二个 $\mathrm{pointwise}$ 卷积后的 $\mathrm{ReLU}$ 激活函数

  <center>
  <img src="images/shufflenet_v1.png"/>
  </center>

### 网络结构

- 第一层是步长为 $2$ 的 $3 \times 3$ 卷积

- 第二层是步长为 $2$ 的 $3 \times 3 \ \mathrm{max-pooling}$

- 最后一层通过 $\mathrm{global \ average \ pooling}$ 后进行 $\mathrm{softmax}$ 分类

- 其余层分为 $3$ 组，每组包含若干个 $\mathrm{ShuffleNet}$ 单元

### 缩放因子 $t$

- 将每一层的通道数缩放为标准 $\mathrm{ShuffleNet}$ 的 $t$ 倍，可以得到不同的模型

### 性能分析

- 假设 $\mathrm{bottleneck}$ 的输入为 $c \times h \times w$，中间层通道数为 $m$，输入特征图分为 $g$ 组

- $\mathrm{ResNet}$ 残差块的计算量如下：

  $$
  h \cdot w \cdot \left( 2 \cdot c \cdot m + 9 \cdot m^{2} \right)
  $$

  - 第一个 $1 \times 1$ 卷积计算量：

    $$
    h \cdot w \cdot c \cdot m
    $$

  - $3 \times 3$ 卷积计算量：

    $$
    h \cdot w \cdot m \cdot m \cdot 3 \cdot 3
    $$

  - 第二个 $1 \times 1$ 卷积计算量：

    $$
    h \cdot w \cdot m \cdot c
    $$

- $\mathrm{ShuffleNet}$ 单元的计算量如下：

  $$
  h \cdot w \cdot \left( \frac{2 \cdot c \cdot m}{g} + 9 \cdot m \right)
  $$

  - 第一个 $\mathrm{pointwise}$ 分组卷积计算量：

    $$
    h \cdot w \cdot g \cdot \left( \frac{c}{g} \cdot \frac{m}{g} \right)
    $$

  - $\mathrm{depthwise}$ 卷积计算量：

    $$
    h \cdot w \cdot g \cdot \left( \frac{m}{g} \cdot 3 \cdot 3 \right)
    $$

  - 第二个 $\mathrm{pointwise}$ 分组卷积计算量：

    $$
    h \cdot w \cdot g \cdot \left( \frac{m}{g} \cdot \frac{c}{g} \right)
    $$

- 同等的计算量下，$\mathrm{ShuffleNet}$ 比 $\mathrm{ResNet}$ 更宽；对于小网络而言，提取的特征更充分

  - 在一定程度上，使用较大的 $g$ 可以抵消 $m$ 增加带来的计算量

- 同等的计算量下，在一定的范围内，分组越多，模型准确率越高

- 实验结果表明，$g = 3$ 时能较好的平衡速度和准确率之间的关系

## $\mathrm{ShuffleNet \ v2}$

- $\mathrm{FLOPS}$ 并不是衡量模型性能的直接指标，应该直接使用速度作为指标

  - 内存访问、并行资源等限制，都会影响实际速度

  - 平台不同时，实际速度也不同

### 设计思想

#### 输入输出通道相同，内存访问次数最少

- 对于 $\mathrm{pointwise}$ 卷积，特征图维度为 $h \times w$，输入输出通道数分别为 $c_{1}, \ c_{2}$，总计算量 $ B = hwc_{1}c_{2} $

- 假设 $\mathrm{cache}$ 可以存储所有特征图和参数，内存访问总次数计算如下：

  $$
  \mathrm{MAC} = hw \left( c_{1} + c_{2} \right) + c_{1} c_{2}
  $$

  - 第一项是读写特征图的 $\mathrm{IO}$ 次数

  - 第二项是读卷积核的 $\mathrm{IO}$ 次数（$c_{1} c_{2}$ 个卷积核）

- 根据均值不等式 $a + b \ge 2 \sqrt{ab}$：

  $$
  \mathrm{MAC} \ge 2hw \sqrt{c_{1} c_{2}} + c_{1} c_{2} = 2 \sqrt{hwB} + \frac{B}{hw}
  $$

  - 当 $c_{1} = c_{2}$，即输入、输出通道数相同时，内存访问次数最少

#### 卷积分组增加，内存访问次数也增加

- 对于 $g$ 组的 $\mathrm{pointwise}$ 分组卷积，总的计算量如下：

  $$
  B = \frac{hw c_{1} c_{2}}{g}
  $$

- 内存访问总次数计算如下：

  $$
  \mathrm{MAC} = hw \left( c_{1} + c_{2} \right) + \frac{c_{1} c_{2}}{g}
  $$

- 运算量不变时，增加卷积分组数意味着增加输入输出通道数，导致内存访问次数增加

#### 网络分支增多，实际速度会变慢

- 网络分支增多时，在一定程度上可以提高模型的准确率

- 串行分支（堆叠层数）过多，每层通道数较少，无法充分利用 $\mathrm{GPU}$ 并行能力

- 并行分支过多，内核启动和内核同步所需时间较长，

#### $\mathrm{Element-wise}$ 操作的时间不可忽视

- 此处的 $\mathrm{element-wise}$ 操作泛指 $\frac{\mathrm{MAC}}{\mathrm{FLOPs}}$ 较高的操作，除了 $\mathrm{ReLU}$、$\mathrm{Add}$、$\mathrm{Bias}$ 等操作外，还包括 $\mathrm{depthwise}$ 卷积

- 对于 $\mathrm{ResNet}$ 的 $\mathrm{bottleneck}$ 结构，移除 $\mathrm{ReLU}$ 和 $\mathrm{shortcut}$ 后，速度上有 $20\%$ 的提升

### $\mathrm{ShuffleNet}$ 单元

- 当 $\mathrm{depthwise}$ 卷积步长为 $1, \ 2$ 时，$\mathrm{shufflenet}$ 单元的结构分别如下：

  - 步长为 $1$ 时，输入通道分为两组，输出通道数不变

    - 卷积分支依次通过 $\mathrm{pointwise}$、$\mathrm{depthwise}$、$\mathrm{pointwise}$ 卷积提取特征，$3$ 层通道数始终相同

    - 恒等映射分支与卷积分支级联后，经过 $\mathrm{channel \ shuffle}$ 操作，进入下一个 $\mathrm{shufflenet}$ 单元

  - 步长为 $2$ 时，不对输入进行拆分，输出通道数加倍

    - 一个卷积分支使用与步长为 $1$ 时相同的结构

    - 另一个分支也采用类似结构，但不使用第一个 $\mathrm{pointwise}$ 卷积

  - 去掉 $\mathrm{depthwise}$ 卷积后的 $\mathrm{ReLU}$ 激活函数

  <center>
  <img src="images/shufflenet_v2.png"/>
  </center>

- 工程上，还可以把 $\mathrm{concat}$、$\mathrm{channel \ shuffle}$ 以及下一个单元的 $\mathrm{channel \ split}$ 操作合并到一起，减少 $\mathrm{elment-wise}$ 操作带来的时间消耗

- 整体的网络结构与 $\mathrm{ShuffleNet \ v1}$ 基本相同，区别在于 $\mathrm{global \ average \ pooling}$ 前增加了一层 $\mathrm{pointwise}$ 卷积

### 网络性能

- 同等运算量下，$\mathrm{ShuffleNet \ v2}$ 比 $\mathrm{MobileNet}$ 速度更快，精度更高

  - $\mathrm{ShuffleNet \ v2}$ 设计出发点就是提升同等运算量下的实际速度

  - $\mathrm{channel \ split}$ 的特征复用基本不增加运算量，$\mathrm{ShuffleNet \ v2}$ 可以使用更多特征图