# 概率论基础

## 期望

- 随机变量 $x$ 的期望计算如下：

  - 离散值：

    $$
    \mathbb{E} \left[ x \right] = \sum_{x} P \left( x \right) \cdot x
    $$

  - 连续值：

    $$
    \mathbb{E} \left[ x \right] = \int P \left( x \right) \cdot x \mathrm{d}x
    $$

- 线性运算的期望等于期望的线性运算：

  $$
  \mathbb{E} \left( \sum_{i = 1}^{n} a_{i} x_{i} + c \right) = \sum_{i = 1}^{n} a_{i} \mathbb{E} \left( x_{i} \right) + c
  $$

- 如果两个变量相互独立，乘积的期望等于期望的乘积：

  $$
  \mathbb{E} \left( xy \right) = \mathbb{E} \left( x \right) \cdot \mathbb{E} \left( y \right)
  $$

## 方差

- 随机变量 $x$ 的方差计算如下：

  - 离散值：

    $$
    \mathrm{Var} \left( x \right) = \mathbb{E} \left[ \left( x - \mathbb{E} \left[ x \right] \right)^{2} \right]
    $$

    - 进一步展开可得：

      $$
      \mathrm{Var} \left( x \right) = \mathbb{E} \left[ x^{2} \right] - \mathbb{E}^{2} \left[ x \right]
      $$

  - 连续值：

    $$
    \mathrm{Var} \left( x \right) = \int P \left( x \right) \cdot \left( x - \mathbb{E} \left[ x \right] \right)^{2} \mathrm{d}x
    $$

    - 进一步展开可得：

      $$
      \mathrm{Var} \left( x \right) = \mathbb{E} \left[ x^{2} \right] - \mathbb{E}^{2} \left[ x \right]
      $$

- 线性运算的方差，等于所有协方差的线性运算：

  $$
  \mathrm{Var} \left( \sum_{i = 1}^{n} a_{i} x_{i} \right) = \sum_{i = 1}^{n} \sum_{j = 1}^{n} a_{i} a_{j} \mathrm{Cov} \left( x_{i}, \ x_{j} \right)
  $$

- 如果两个变量相互独立，线性运算的方差计算如下：

  $$
  \mathrm{Var} \left( ax + by \right) = a^{2} \mathrm{Var} \left( x \right) + b^{2} \mathrm{Var} \left( y \right)
  $$

- 如果两个变量相互独立，且均值为 $0$，乘积的方差计算如下：

  $$
  \mathrm{Var} \left( xy \right) = \mathrm{Var} \left( x \right) \cdot \mathrm{Var} \left( y \right)
  $$

## 条件概率链式法则

- 多维随机变量的联合概率分布，可以分解成只有一个变量的条件概率相乘的形式：

  $$
  P \left( a, \ b, \ c \right) = P \left( a|b, \ c \right) \cdot P \left( b|c \right) \cdot P \left( c \right)
  $$

  - 证明如下：

    $$
    P \left( a, \ b, \ c \right) = P \left( a|b, \ c \right) \cdot P \left( b, \ c \right) = P \left( a|b, \ c \right) \cdot P \left( b|c \right) \cdot P \left( c \right)
    $$

## 常用概率分布

### 伯努利分布

- 随机变量 $x$ 的取值集合为 $\left\{ 0, \ 1 \right\}$，对应概率分别为 $1 - \phi, \ \phi$

  $$
  P \left( x \right) = \phi^{x} \cdot \left( 1 - \phi \right)^{1 - x}
  $$

- 期望：

  $$
  \mathbb{E} \left[ x \right] = 1 \cdot \phi + 0 \cdot \left( 1 - \phi \right) = \phi
  $$

- 方差：

  $$
  \mathrm{Var} \left( x \right) = \mathbb{E} \left[ x^{2} \right] - \mathbb{E}^{2} \left[ x \right] = \phi \left( 1 - \phi \right)
  $$

  - 其中 $\mathbb{E} \left[ x^{2} \right]$ 计算如下：

    $$
    \mathbb{E} \left[ x^{2} \right] = 1^{2} \cdot \phi + 0^{2} \cdot \left( 1 - \phi \right) = \phi
    $$

### 均匀分布

- 随机变量 $x$ 落在区间 $\left[ a, \ b \right]$ 上任一点的概率均为 $\frac{1}{b - a}$

- 期望：

  $$
  \mathbb{E} \left[ x \right] = \int_{a}^{b} x \cdot \frac{1}{b - a} \mathrm{d} x = \frac{a + b}{2}
  $$

- 方差：

  $$
  \mathrm{Var} \left( x \right) = \mathbb{E} \left[ x^{2} \right] - \mathbb{E}^{2} \left[ x \right] = \frac{\left( a - b \right)^{2}}{12}
  $$

  - 其中 $\mathbb{E} \left[ x^{2} \right]$ 计算如下：

    $$
    \mathbb{E} \left[ x^{2} \right] = \int_{a}^{b} P \left( x \right) \cdot x^{2} \mathrm{d} x = \frac{a^{2}+ b ^{2} + ab}{3}
    $$

### 高斯分布

- 又称正态分布，参考 [$\mathrm{GaussianModels.md}$](GaussianModels.md)

### 泊松分布

- 描述单位时间（或空间）内随机事件发生次数的概率分布

- 假设单位时间内事件 $x$ 发生 $\lambda$ 次，则单位时间内，事件 $x$ 发生 $k$ 次的概率为：

  $$
  P \left( x = k \right) = \frac{\lambda^{k} \cdot \mathrm{e}^{-\lambda}}{k!}
  $$

- 在 $t$ 个单位时间内，事件 $x$ 发生 $k$ 次的概率为：

  $$
  P \left( x = k \right) = \frac{ \left( \lambda t \right)^{k} \cdot \mathrm{e}^{-\lambda t}}{k!}
  $$

- 期望、方差均为 $\lambda$

  - 计算时用到 $\mathrm{e}^{x}$ 在 $x = 0$ 处的泰勒展开式

- 泊松分布是非对称分布，当 $\lambda$ 较小时呈偏态分布；当 $\lambda = 20$ 时，近似为高斯分布

## 大数定律

- 对于 $n$ 个独立同分布的样本，当 $n \rightarrow \infty$ 时，样本均值收敛到期望值：

  $$
  \lim_{n \rightarrow \infty}{\frac{1}{n} \sum_{i = 1}^{n} x_{i}} = \mu
  $$

## 中心极限定理

- 对于 $n$ 个独立同分布的样本，当 $n \rightarrow \infty$ 时，样本均值近似服从正态分布：

  $$
  \lim_{n \rightarrow \infty} \frac{1}{n} \sum_{i = 1}^{n} x_{i} \ \sim \ N \left( \mu, \ \frac{\sigma^{2}}{n} \right)
  $$

  - 其中 $\mu$ 是原始样本均值，$\sigma^{2}$ 是原始样本方差

  - 归一化得到：

    $$
    \lim_{n \rightarrow \infty} \frac{\frac{1}{n} \sum_{i = 1}^{n} x_{i} - \mu}{\sigma \ / \sqrt{n}} \ \sim \ N \left( 0, \ 1 \right)
    $$

## 马尔可夫不等式

- 对于随机变量 $X$ 与给定值 $a$：

  $$
  P \left( X \geq a \right) \leq \frac{\mu}{a}
  $$

  - 越大于平均值，概率越低

## 切比雪夫不等式

- 对于随机变量 $X$ 与给定值 $\epsilon$：

  $$
  P \left( |X - \mu| \geq \epsilon \right) \leq \frac{\sigma^{2}}{\epsilon^{2}}
  $$

  - 越偏离平均值，概率越低