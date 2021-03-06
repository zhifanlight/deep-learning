# 正则化

## 背景介绍

- 对训练模型的权值项进行约束，以防止过拟合，保证更好的泛化性能

## 常用方法

### $L_{0}$ 正则化

$$
||w||_{0} = \sum_{i = 1}^{N} I \left( w_{i} \neq 0 \right)
$$

- 通过 $L_{0}$ 范数进行约束，使向量中大部分元素都是 $0$，即权值稀疏

- 由于 $L_{0}$ 范数难以优化求解，通常用其最优凸近似 $L_{1}$ 范数代替

### $L_{1}$ 正则化

$$
||w||_{1} = \sum_{i = 1}^{N} |w_{i}|
$$

- 通过 $L_{1}$ 范数进行约束，使向量中大部分元素都是 $0$，即权值稀疏

- 导数固定，在使用梯度下降求解时：

  - 如果 $w_{i} < 0$，导数为 $-1$，$w_{i}$ 减去 $-\lambda$ 使 $w_{i}$ 变大

  - 如果 $w_{i} > 0$，导数为 $1$，$w_{i}$ 减去 $\lambda$ 使 $w_{i}$ 变小

  - 由于 $w_{i}$ 改变量始终不变，最终向量中大部分元素是 $0$

- 使用 $L_{1}$ 正则化时，假设参数先验服从拉普拉斯分布：

  $$
  p \left( x \right) = \frac{1}{2b} \exp \left( -\frac{|x - \mu|}{b} \right)
  $$

  - 通过最大后验估计，可推导出 $L_{1}$ 正则化

### $L_{2}$ 正则化

$$
||w||_{2} = \sum_{i = 1}^{N} w_{i}^{2}
$$

- 通过 $L_{2}$ 范数进行约束，使向量中大部分元素都接近 $0$，即权值衰减

- 导数为 $2 w_{i}$，在使用梯度下降求解时：

  - 如果 $w_{i} < 0$，导数为负，$w_{i}$ 减去 $2 \lambda w_{i}$ 使 $w_{i}$ 变大

  - 如果 $w_{i} > 0$，导数为正，$w_{i}$ 减去 $2 \lambda w_{i}$ 使 $w_{i}$ 变小

  - 当 $|w_{i}| \rightarrow 0$ 时，$w_{i}$ 改变量越来越小，最终向量中大部分元素都接近 $0$

- 使用 $L_{1}$ 正则化时，假设参数先验服从高斯分布：

  $$
  p \left( x \right) = \frac{1}{\sqrt{2\pi} \sigma} \exp \left( -\frac{ \left( x - \mu \right)^{2}}{2 \sigma^{2}} \right)
  $$

  - 通过最大后验估计，可推导出 $L_{2}$ 正则化

### $\mathrm{max-norm}$ 正则化

$$
||w||_{2} \leq c
$$

- $L_{2}$ 范数不超过 $c$，$w$ 被限制在半径为 $c$ 的超球体内

- 当 $||w||_{2} > c$ 时，对每一维等比例缩放，以保证 $||w||_{2} = c$

- 在一定程度上防止梯度爆炸

## 向量范数

### $0$ 范数

- 向量中非零元素个数：

  $$
  ||x||_{0} = \sum_{i = 1}^{N} I \left( x_{i} \neq 0 \right)
  $$

### $1$ 范数

- 向量元素绝对值之和：

  $$
  ||x||_{1} = \sum_{i = 1}^{N} |x_{i}|
  $$

### $2$ 范数（$\mathrm{Euclid}$ 范数）

- 向量元素平方和的平方根：

  $$
  ||x||_{2} = \sqrt{\sum_{i = 1}^{N} x_{i}^{2}}
  $$

### $\infty$ 范数

- 向量元素绝对值中最大值：

  $$
  ||x||_{\infty} = \max |x_{i}|
  $$

### $-\infty$范数

- 向量元素绝对值中最小值：

  $$
  ||x||_{-\infty} = \min |x_{i}|
  $$

### $p$ 范数

- 向量元素绝对值 $p$ 次方和的 $p$ 次方根：

  $$
  ||x||_{p} = \left( \sum_{i = 1}^{N} |x_{i}|^{p} \right)^{\frac{1}{p}}
  $$