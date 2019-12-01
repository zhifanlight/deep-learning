# 朴素贝叶斯

## 基本思想

- 假设输入向量各个维度之间相互独立

- 通过学习联合分布 $P \left( X, \ Y \right)$ 来学习条件概率 $P \left( Y|X \right)$

## 数学推导

### 基本方法

- 假设输入为 $N$ 维向量 $x$，输出为标量 $y \in \left\{c_{1}, \ c_{2}, \ \cdots, c_{K} \right\}$

- 条件概率计算如下：

  $$
  P \left( X = x | Y = c_{k} \right) = P \left( X_{1} = x_{1}, \ X_{2} = x_{2}, \ \cdots \ X_{N} = x_{N} | Y = c_{k} \right)
  $$

  - 假设 $x_{j}$ 的取值有 $S_{j}$ 个，模型的参数个数为：

    $$
    K \prod_{j = 1}^{N} S_{j}
    $$

- 如果假设输入的各个维度间相互独立，条件概率计算如下：

  $$
  P \left( X = x | Y = c_{k} \right) = \prod_{j = 1}^{N} P \left( X_{j} = x_{j} | Y = c_{k} \right)
  $$

  - 模型的参数个数为：

    $$
    K \sum_{j = 1}^{N} S_{j}
    $$

- 朴素贝叶斯后验概率计算如下：

  $$
  P \left( Y = c_{k} | X = x \right) = \frac{P \left( X = x, \ Y = c_{k} \right)}{P \left( X = x \right)}
  $$

  - 当 $x$ 固定时，$P \left( X = x \right)$ 为常量；因此后验概率计算如下：

    $$
    \begin{aligned}
    P \left( Y = c_{k} | X = x \right) &\propto P \left( X = x, \ Y = c_{k} \right) \newline
    &= P \left( Y = c_{k} \right) \cdot P \left( X = x | Y = c_{k} \right) \newline
    &= P \left( Y = c_{k} \right) \prod_{j = 1}^{N} P \left( X_{j} = x_{j} | Y = c_{k} \right) \newline
    \end{aligned}
    $$

  - 选择最大 $P \left( Y = c_{k} | X = x \right)$ 对应的 $c_{k}$ 作为最终类别标签

### 参数估计

- 先验概率：

  $$
  P \left( Y = c_{k} \right) = \frac{\sum_{i = 1}^{M} I \left( y_{i} = c_{k} \right)}{M}
  $$

- 条件概率：

  $$
  P \left( X_{j} = a_{t} | Y = c_{k} \right) = \frac{\sum_{i = 1}^{M} I \left( x_{ij} = a_{t}, \ y_{i} = c_{k} \right)}{\sum_{i = 1}^{M} \left( y_{i} = c_{k} \right)}
  $$

### 拉普拉斯平滑

- 采用朴素贝叶斯估计时，某些概率值可能为 $0$

- 先验概率计算如下：

  $$
  P \left( Y = c_{k} \right) = \frac{\lambda + \sum_{i = 1}^{M} I \left( y_{i} = c_{k} \right)}{K \lambda + M}
  $$

- 条件概率计算如下：

  $$
  P \left( X_{j} = a_{t} | Y = c_{k} \right) = \frac{\lambda + \sum_{i = 1}^{M} I \left( x_{ij} = a_{t}, \ y_{i} = c_{k} \right)}{|S_{j}| \cdot \lambda + \sum_{i = 1}^{M} \left( y_{i} = c_{k} \right)}
  $$

  - 其中，$S_{j}$ 是 $x_{j}$ 的取值集合

- $\lambda$ 通常为 $1$