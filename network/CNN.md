# $\mathrm{Convolutional \ Neural \ Networks}$

## 背景介绍

- 与 $\mathrm{BP}$ 网络相比，$\mathrm{CNN}$ 的隐藏层细分为卷积层、$\mathrm{Pooling}$ 层、全连接层

- 输入是二维图像而非一维向量，因此可以学到像素点的空间关系

- 卷积层、$\mathrm{Pooling}$ 层的梯度计算方式与 $\mathrm{BP}$ 的全连接层不同，但训练过程相同

<center>
<img src="images/lenet.png"/>
</center>

### 卷积层

- 输入图像与不同的卷积核做卷积，得到不同的特征图像

- 在训练时，需要进行卷积核、偏置项等参数的学习

- 输入尺寸 $i$、输出尺寸 $o$、$\mathrm{padding}$ 长度 $p$、卷积核大小 $k$、步长 $s$ 的关系为：

  $$
  o = \left \lfloor \frac{i - k + 2p}{s} \right \rfloor + 1
  $$

- $1 \times 1$ 卷积核的作用：

  - 整合不同通道的信息

  - 实现通道层面的降维，以减少计算

### $\mathrm{Pooling}$ 层

- 降低特征图像的维度，分为 $\mathrm{max-pooling}$ 和 $\mathrm{average-pooling}$

  - $\mathrm{max-pooling}$

    - 取邻域最大值作为当前像素点的像素值

  - $\mathrm{average-pooling}$

    - 取邻域平均值作为当前像素点的像素值

- 卷积层生成的特征图通常包含冗余信息，需要通过 $\mathrm{Pooling}$ 来去除冗余

  - 如果使用 $\mathrm{max-pooling}$，相当于只考虑最大激活值

  - 而使用 $\mathrm{average-pooling}$，相当于考虑了所有冗余信息

- 通常使用 $\mathrm{max-pooling}$；但分类时的最后一个池化层可以进行 $\mathrm{global \ average \ pooling}$

- 没有参数，在训练时无需学习

- 输入尺寸 $i$、输出尺寸 $o$、池化核大小 $k$、步长 $s$ 的关系为：

  $$
  o = \left \lceil \frac{i - k}{s} \right \rceil + 1
  $$

## 数学推导

### 单通道输入、单通道特征图

#### 卷积层

- 定义如下：

  - $w_{m, \ n}^{l} \quad$：第 $l$ 层的卷积核权值

  - $b^{l} \quad$：第 $l$ 层的偏置

  - $E \quad$：输出层误差

  - $x_{i, \ j}^{l} \quad$：第 $l$ 层 $\left( i, \ j \right)$ 像素点的加权输入

    $$
    x_{i, \ j}^{l} = \sum_{m} \sum_{n} w_{m, \ n}^{l} \cdot z_{i + m, \ j + n}^{l - 1} + b^{l}
    $$

  - $z_{i, \ j}^{l} \quad$：第 $l$ 层 $\left( i, \ j \right)$ 像素点的预测输出

    $$
    z_{i,\ j}^{l} = f \left( x_{i, \ j}^{l} \right)
    $$

  - $\delta_{i, \ j}^{l} \quad$：第 $l$ 层 $\left( i, \ j \right)$ 像素点的误差项

    $$
    \delta_{i, \ j}^{l} = \frac{\partial{E}}{\partial{x_{i, \ j}^{l}}}
    $$

- 由梯度下降可得：

  $$
  \left\{ \begin{matrix}
  w_{m, \ n}^{l} \leftarrow w_{m, \ n}^{l} - \eta \cdot \frac {\partial{E}}{\partial{w_{m, \ n}^{l}}} \\
  b^{l} \leftarrow b^{l} - \eta \cdot \frac {\partial{E}}{\partial{b^{l}}}
  \end{matrix} \right.
  $$

- 对权重项应用链式法则：

  $$
  \frac{\partial{E}}{\partial{w_{m, \ n}^{l}}} = \sum_{i} \sum_{j}\frac{\partial{E}}{\partial{x_{i, \ j}^{l}}} \cdot \frac{\partial{x_{i, \ j}^{l}}}{\partial{w_{m, \ n}^{l}}} = \sum_{i} \sum_{j} \delta_{i, \ j}^{l} \cdot z_{i + m, \ j + n}^{l - 1}
  $$

  - 于是 $w_{ji}$ 更新公式为：

    $$
    w_{m, \ n}^{l} \leftarrow w_{m, \ n}^{l} - \eta \cdot \sum_{i} \sum_{j} \delta_{i, \ j}^{l} \cdot z_{i + m, \ j + n}^{l - 1}
    $$

- 对偏置项应用链式法则：

  $$
  \frac{\partial{E}}{\partial{b^{l}}} = \sum_{i} \sum_{j}\frac{\partial{E}}{\partial{x_{i, \ j}^{l}}} \cdot \frac{\partial{x_{i, \ j}^{l}}}{\partial{b^{l}}} = \sum_{i} \sum_{j} \delta_{i, \ j}^{l}
  $$

  - 于是：

    $$
    b^{l} \leftarrow b^{l} - \eta \cdot \sum_{i} \sum_{j} \delta_{i, \ j}^{l}
    $$

- 误差项 $ \delta_{i, \ j}^{l} = \frac{\partial{E}}{\partial{x_{i, \ j}^{l}}} $ 计算如下：

  - 假设受输入 $x_{i, \ j}^{l}$ 影响的区域为 $Q$，则：

    $$
    \frac{\partial{E}}{\partial{x_{i, \ j}^{l}}} = \sum_{Q} \frac{\partial{E}}{\partial{x_{Q}^{l + 1}}} \cdot \frac{\partial{x_{Q}^{l + 1}}}{\partial{x_{i, \ j}^{l}}}
    $$

  - 假设 $X = \left[ \begin{matrix} a & b & c \\ d & e & f \\ g & h & j \end{matrix} \right]$，$W = \left[ \begin{matrix} w & x \\ y & z \end{matrix} \right]$，$Y = \left[ \begin{matrix} p & q \\ r & s \end{matrix} \right]$，$\delta_{X} = \left[ \begin{matrix} \delta_{a} & \delta_{b} & \delta_{c} \\ \delta_{d} & \delta_{e} & \delta_{f} \\ \delta_{g} & \delta_{h} & \delta_{j} \end{matrix} \right]$，$\delta_{Y} = \left[ \begin{matrix} 0 & 0 & 0 & 0 \\ 0 & \delta_{p} & \delta_{q} & 0 \\ 0 & \delta_{r} & \delta_{s} & 0 \\ 0 & 0 & 0 & 0 \end{matrix} \right]$

- 正向计算时：

  $$
  Y = X \otimes W
  $$

  - 由卷积定义：

    $$
    \left\{ \begin{matrix} p = aw + bx + dy + ez \\ q = bw + cx + ey + fz \\ r = dw + ex + gy + hz \\ s = ew + fx + hy + jz \end{matrix} \right.
    $$

- 反向传播时：

  $$
  \delta_{X} = \delta_{Y} \otimes \mathrm{Rot} \left( W \right)
  $$

  - 其中 $\mathrm{Rot} \left( W \right) = \left[ \begin{matrix} z & y \\ x & w \end{matrix} \right]$ 表示对卷积核 $W$ 同时进行水平翻转和竖直翻转

  - 以 $\delta_{a}, \ \delta_{e}$ 为例，偏导数计算如下：

    $$
    \frac{\partial{E}}{\partial{a}} = \frac{\partial{E}}{\partial{p}} \cdot \frac{\partial{p}}{\partial{a}} = \delta_{p} \cdot w
    $$

    $$
    \frac{\partial{E}}{\partial{e}} = \frac{\partial{E}}{\partial{p}} \frac{\partial{p}}{\partial{e}} + \frac{\partial{E}}{\partial{q}} \frac{\partial{q}}{\partial{e}} + \frac{\partial{E}}{\partial{r}} \frac{\partial{r}}{\partial{e}} + \frac{\partial{E}}{\partial{s}} \frac{\partial{s}}{\partial{e}} = \delta_{p} \cdot z + \delta_{q} \cdot y + \delta_{r} \cdot x + \delta_{s} \cdot w
    $$

- 在实际计算时，由于进行了 $\mathrm{im2col}$ 和 $\mathrm{col2im}$，不用翻转卷积核；但需要对 $\mathrm{im2col}$ 后的卷积核进行转置

  - 正向计算：

    - 进行 $\mathrm{im2col}$：

      $$
      W = \left[ \begin{matrix} w & x & y & z \end{matrix} \right], \quad X = \left[ \begin{matrix} a & b & d & e \\ b & c & e & f \\ d & e & g & h \\ e & f & h & j \end{matrix} \right]
      $$

    - 进行矩阵乘法：

      $$
      Y = WX = \left[ \begin{matrix} wa + xb + yd + ze \\ wb + xc + ye + zf \\ wd + xe + yg + zh \\ we + xf + yh + zj \end{matrix} \right]^{T}
      $$

  - 反向传播：

    - 误差矩阵不使用 $\mathrm{padding}$：

      $$
      \delta_{Y} = \left[ \begin{matrix} \delta_{p} & \delta_{q} \\ \delta_{r} & \delta_{s} \end{matrix} \right]
      $$

    - 进行 $\mathrm{im2col}$：

      $$
      W^{T} = \left[ \begin{matrix} w \\ x \\ y \\ z \end{matrix} \right], \quad \delta_{Y} = \left[ \begin{matrix} \delta_{p} & \delta_{q} & \delta_{r} & \delta_{s} \end{matrix} \right]
      $$

    - 进行矩阵乘法：

      $$
      \delta_{X} = W^{T} \delta_{Y} = \left[ \begin{matrix} w \delta_{p} & w \delta_{q} & w \delta_{r} & w \delta_{s} \\ x \delta_{p} & x \delta_{q} & x \delta_{r} & x \delta_{s} \\ y \delta_{p} & y \delta_{q} & y \delta_{r} & y \delta_{s} \\ z \delta_{p} & z \delta_{q} & z \delta_{r} & z \delta_{s} \end{matrix} \right]
      $$

    - 对 $\delta_{X}$ 先转置再进行 $\mathrm{col2im}$：

      $$
      \delta_{X} = \left[ \begin{matrix} w \delta_{p} & w \delta_{q} + x \delta_{p} & x \delta_{q} \\ w \delta_{r} + y \delta_{p} & w \delta_{s} + x \delta_{r} + y \delta_{q} + z \delta_{p} & x \delta_{s} + z \delta_{q} \\ y \delta_{r} & y \delta_{s} + z \delta_{r} & z \delta_{s} \end{matrix} \right]
      $$

#### $\mathrm{Pooling}$ 层

- $\mathrm{max-pooling}$

  - 根据前向计算时记录的最大值位置，将误差原封不动地传到前一层邻域最大值处

- $\mathrm{average-pooling}$

  - 将后一层误差均匀地传到前一层邻域内的所有像素点