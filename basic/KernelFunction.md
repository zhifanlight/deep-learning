# 核函数

## 背景介绍

- 在原始空间线性不可分的问题在高维空间可分，从低维空间变换到高维空间需要映射

- 核函数可以用于 $\mathrm{SVM}$ 中，但不仅限于用在 $\mathrm{SVM}$ 中

## 数学推导

- 假设 $\phi \left( x \right)$ 表示原始空间向高维空间的映射

- 在使用核函数时，通常具有 $x_{i}^{T} x_{j}$ 的形式，用 $\phi \left( x_{i} \right), \ \phi \left( x_{j} \right)$ 分别替代 $x_{i}, \ x_{j}$ 即可实现向高维空间的映射

- 由于 $\phi \left( x \right)$ 是高维甚至无穷维，计算不便，可以通过引入核函数 $K$ 直接计算乘积 $\phi \left( x_{i} \right)^{T} \phi \left( x_{j} \right)$：

  $$
  K \left( x_{i}, \ x_{j} \right) = \phi \left( x_{i} \right)^{T} \phi \left( x_{j} \right)
  $$

## 常用核函数

### 多项式核

$$
K \left( x_{i}, \ x_{j} \right) = \left( x_{i}^{T} x_{j} \right)^{d} \qquad d \geq 1
$$

- 多项式核将数据从原始空间映射到更高维：

  - 以 $K \left( x, \ y \right) = \left( x^{T} y \right)^{2} $ 对二维向量 $\left( x, \ y \right)$ 进行映射为例，展开可得：

  $$
  K \left( x, \ y \right) = \left( x_{1} x_{2} + y_{1} y_{2} \right)^{2} = x_{1}^{2} x_{2}^{2} + 2 x_{1} x_{2} y_{1} y_{2} + y_{1}^{2} y_{2}^{2}
  $$

  - 此时特征映射函数为：

    $$
    \Phi \left( x, \ y \right) = \left( x^{2}, \ \sqrt{2}xy, \ y^{2} \right)
    $$

    - 实现了二维空间到三维空间的映射

### 线性核

$$
K \left( x_{i}, \ x_{j} \right) = x_{i}^{T} x_{j}
$$

### 高斯核（$\mathrm{RBF}$ 核）

$$
K \left( x_{i}, \ x_{j} \right) = \exp \left( -\frac{||x_{i} - x_{j}||^{2}}{2 \sigma^{2}} \right) \qquad \sigma > 0
$$

- 高斯核将数据从原始空间映射到无穷维：

  - 高斯核带宽 $\sigma$ 不影响数据维度，对高斯核进行化简可得：

    $$
    K \left( x, \ y \right) = \exp \left( -||x||^{2} \right) \cdot \exp \left( -||y||^{2} \right) \cdot \exp \left( 2x^{T} y \right)
    $$

  - 根据 $f \left( x \right)$ 在 $x = 0$ 处泰勒公式可得：

    $$
    K \left( x, \ y \right) = \exp \left( -||x||^{2} \right) \cdot \exp \left( -||y||^{2} \right) \cdot \sum_{n = 0}^{+\infty} \frac{\left( 2x^{T} y \right)^{n}}{n!}
    $$

    - 高斯核本质是无穷多个多项式核累加

    - 由多项式核将数据映射到更高维可知，高斯核可以将数据映射到无穷维

### 拉普拉斯核

$$
K \left( x_{i}, \ x_{j} \right) = \exp \left( -\frac{||x_{i} - x_{j}||}{\sigma} \right) \qquad \sigma > 0
$$

### $\mathrm{Sigmoid}$ 核

$$
K \left( x_{i}, \ x_{j} \right) = \tanh \left( \beta x_{i}^{T} x_{j} + \theta \right) \qquad \beta > 0, \ \theta < 0
$$