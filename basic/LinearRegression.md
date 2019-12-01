# 线性回归

## 背景介绍

- 在回归任务中，最基础的模型就是线性回归，即输入值和输出值之间存在简单的线性关系：

  - 在二维平面中，线性是一条直线

  - 在三维空间中，线性是一个平面

  - 在多维空间中，线性是一个超平面

- 对于线性回归任务，假设获得了 $N$ 维空间中的 $M$ 个点及观察值：$\left( x_{1}, \ y_{1} \right), \ \left( x_{2}, \ y_{2} \right), \ \cdots, \ \left( x_{M}, \ y_{M} \right)$，其中 $x_{i}$ 是 $N$ 维列向量 $\left[ \begin{matrix} x_{i1} \\ x_{i2} \\ \vdots \\ x_{iN} \end{matrix} \right]$，$y_{i}$ 是标量

- 对于这些点，最好的拟合超平面要求总的误差最小，有以下三个标准可以选择：

  - 误差和最小：容易使得正、负误差互相抵消

  - 误差绝对值和最小：难以优化求解

  - 误差平方和最小：计算方便，符合欧式距离

- 使用 $L_{1}$ 正则项时称为 $\mathrm{Lasso \ Regression}$，使用 $L_{2}$ 正则项时称为 $\mathrm{Ridge \ Regression}$

- $\mathrm{Lasso \ Regression}$ 和 $\mathrm{Ridge \ Regression}$ 均可通过最大后验估计推导：

  - 假设参数服从拉普拉斯分布时，可以推导出 $\mathrm{Lasso \ Regression}$

  - 假设参数服从高斯分布时，可以推导出 $\mathrm{Ridge \ Regression}$

## 数学推导

### 模型建立

- 采用误差平方和时，通常使用最小二乘法进行拟合

- 如果把 $N$ 维列向量 $x_{i}$ 扩充到 $N + 1$ 维并令 $x_{i0} = 1$，那么待拟合超平面可以定义为 $h \left( x \right) = \theta^{T} x$，其中 $\theta$ 是 $N + 1$ 维列向量，则损失函数定义如下：

  $$
  J \left( \theta \right) = \frac{1}{2} \sum_{i = 1}^{M} \left( h \left( x_{i} \right) - y_{i} \right)^{2} = \frac{1}{2} \sum_{i = 1}^{M} \left( \theta^{T} x_{i} - y_{i} \right)^{2}
  $$

- 令 $X = \left[ \begin{matrix} x_{10} & x_{11} & \cdots & x_{1N} \\ x_{20} & x_{21} & \cdots x_{2N} \\ \vdots & \vdots & \ddots & \vdots \\ x_{M0} & x_{M1} & \cdots & x_{MN} \end{matrix} \right], \ \theta = \left[ \begin{matrix} \theta_{0} \\ \theta_{1} \\ \vdots \\ \theta_{N} \end{matrix} \right], \ Y = \left[ \begin{matrix} y_{1} \\ y_{2} \\ \vdots \\ y_{M} \end{matrix} \right] $，则 $X \theta - Y = \left[ \begin{matrix} x_{1}^{T} \theta - y_{1} \\ x_{2}^{T} \theta - y_{2} \\ \vdots \\ x_{M}^{T} \theta - y_{M} \end{matrix} \right]$

### 优化求解

#### 梯度下降

- 通过梯度下降法求解 $\theta_{j} \leftarrow \eta \cdot \nabla_{\theta_{j}} J \left( \theta \right)$：

  $$
  \begin{aligned}
  \nabla_{\theta_{j}}J \left( \theta \right) &= \sum_{i = 1}^{M} \left( \theta^{T} x_{i} - y_{i} \right) \cdot \frac{\partial{\left( \theta^{T} x_{i} - y_{i} \right)}}{\partial{\theta_{j}}} \newline
  &= \sum_{i = 1}^{M} \left( \theta^{T} x_{i} - y_{i} \right) \cdot x_{ij} \newline
  &= \sum_{i = 1}^{M} \left( h \left( x_{i} \right) - y_{i} \right) \cdot x_{ij} \newline
  \end{aligned}
  $$

- 向量化如下：

  $$
  \theta \leftarrow \theta - \eta \cdot X^{T} \left( X \theta - Y \right)
  $$

#### 正规方程

- 由于 $X \theta - Y$ 为 $M$ 维列向量：

  $$
  \min_{\theta} J \left( \theta \right) = \min_{\theta} \frac{1}{2} \left( X \theta - Y \right)^{T} \left( X \theta - Y \right) + \frac{\lambda}{2} ||\theta||^{2}
  $$

- 对 $J \left( \theta \right)$ 求导可得：

  $$
  \nabla_{\theta} J \left( \theta \right) = X^{T} X \theta - X^{T}Y + \lambda \theta
  $$

- 令导数为 $0$ 可得最优解：

  $$
  \theta^{*} = \left( X^{T} X + \lambda I \right)^{-1} X^{T} Y
  $$

## 最小二乘概率解释

### 允许误差存在

- 由于所有预测值都不可能完美地与真实值契合，误差必然存在；而拟合的目标就是让误差尽可能小

- 因此，输入值 $x_{i}$ 和输出值 $y_{i}$ 之间的关系可表示为：

  $$
  y_{i} = \theta^{T} x_{i} + \epsilon_{i}
  $$

### 假设误差分布

- 误差项 $\epsilon_{i}$ 捕捉未被设置为特征的变量，假设其独立同分布，且服从高斯分布 $N \left( 0, \ \sigma^{2} \right)$：

  $$
  p \left( \epsilon_{i} \right) = \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(- \frac{\epsilon_{i}^{2}}{2 \sigma^{2}} \right)
  $$

- 在以 $\theta$ 为参数且 $x_{i}$ 给定时，$\theta^{T} x_{i}$ 为定值，此时的 $y_{i}$ 服从高斯分布 $N \left( \theta^{T} x_{i}, \ \sigma^{2} \right)$：

  $$
  p \left( y_{i}|x_{i}; \ \theta \right) = \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(- \frac{ \left( y_{i} - \theta^{T} x_{i} \right)^{2}}{2 \sigma^{2}} \right)
  $$

### 最大似然估计

- 对数似然计算如下：

  $$
  \begin{aligned}
  \log L \left( \theta \right) &= \log \prod_{i = 1}^{M} \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(- \frac{ \left( y_{i} - \theta^{T} x_{i} \right)^{2}}{2 \sigma^{2}} \right) \newline
  &= \sum_{i = 1}^{M} \log \frac{1}{\sqrt{2 \pi} \sigma} \exp \left(- \frac{ \left( y_{i} - \theta^{T} x_{i} \right)^{2}}{2 \sigma^{2}} \right) \newline
  &= M \log \frac{1}{\sqrt{2 \pi} \sigma} - \frac{1}{\sigma^{2}} \cdot \frac{1}{2} \sum_{i = 1}^{M} \left( y_{i} - \theta^{T} x_{i} \right)^{2} \newline
  \end{aligned}
  $$

- 最小化损失函数，实质上是最大化对数似然函数：

  $$
  \min J \left( \theta \right) = \max \log L \left( \theta \right)
  $$