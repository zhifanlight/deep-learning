# $\mathrm{Softmax}$ 回归

## 背景介绍

- $\mathrm{softmax}$ 回归用于多分类任务，其结果是多个超平面

- 对于多分类任务，假设获得了 $N$ 维空间中的 $M$ 个点及观察值：$\left( x_{1}, \ y_{1} \right), \ \left( x_{2}, \ y_{2} \right), \ \cdots, \ \left( x_{M}, \ y_{M} \right)$，其中 $x_{i}$ 是 $N$ 维列向量 $\left[ \begin{matrix} x_{i1} \\ x_{i2} \\ \vdots \\ x_{iN} \end{matrix} \right]$，$y_{i}$ 是标量

## 数学推导

### 模型建立

- 如果把 $N$ 维列向量 $x_{i}$ 扩充到 $N + 1$ 维并令 $x_{i0} = 1$，那么第 $i$ 个待拟合超平面可以定义为 $\theta_{i}^{T }x$，其中 $\theta_{i}$ 是 $N + 1$ 维列向量

- 假设后验概率服从多项式分布，于是：

  $$
  p \left( y|x; \ \theta \right) = \prod_{i = 1}^{K} \phi_{i}^{I \left( y = i \right)}
  $$

  - $\phi_{i}$ 表示 $y = i|x$ 的概率：

    $$
    \phi_{i} = \frac{\mathrm{e}^{\theta_{i}^{T} x}}{\sum_{j = 1}^{K} \mathrm{e}^{\theta_{j}^{T} x}}
    $$

  - $I \left( c \right)$ 是指示函数：

    $$
    I \left( c \right) = \left\{ \begin{matrix} 1, \quad \mathrm{if \ c \ is \ true} \\ 0, \quad \mathrm{if \ c \ is \ false} \end{matrix} \right.
    $$

- 对数似然计算如下：

  $$
  \begin{aligned}
  L \left( \theta \right) &= \log \prod_{i = 1}^{M} \prod_{j = 1}^{K} \left( \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}} \right)^{I \left( y_{i} = j \right)} \newline
  &= \sum_{i = 1}^{M} \sum_{j = 1}^{K} I \left( y_{i} = j \right) \cdot \log \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}} \newline
  \end{aligned}
  $$

- 由于最大化对数似然函数等于最小化损失函数，损失函数计算如下：

  $$
  J \left( \theta \right) = - \sum_{i = 1}^{M} \sum_{j = 1}^{K} I \left( y_{i} = j \right) \cdot \log \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}}
  $$

### 优化求解

#### 梯度下降

- 通过梯度下降法求解 $\theta_{jk} \leftarrow \eta \cdot \nabla_{\theta_{jk}} J \left( \theta \right)$：

  $$
  \begin{aligned}
  \nabla_{\theta_{jk}} J \left( \theta \right) &= - \left( \sum_{i = 1}^{M} I \left( y_{i} = j \right) \cdot \frac{\partial \theta_{j}^{T} x_{i}}{\partial \theta_{jk}} - \sum_{i = 1}^{M} I \left( y_{i} = j \right) \cdot \frac{\partial \log \sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}}{\partial \theta_{jk}} - \sum_{i = 1}^{M} \sum_{c \neq j}^{K} I \left( y_{i} = c \right) \cdot \frac{\partial \log \sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}}{\partial \theta_{jk}} \right) \newline
  &= - \left( \sum_{i = 1}^{M} I \left( y_{i} = j \right) \cdot x_{ik} - \sum_{i = 1}^{M} I \left( y_{i} = j \right) \cdot \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}} \cdot x_{ik} - \sum_{i = 1}^{M} \sum_{c \neq j}^{K} I \left( y_{i} = c \right) \cdot \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}} \cdot x_{ik} \right) \newline
  &= - \left( \sum_{i = 1}^{M} I \left( y_{i} = j \right) \cdot x_{ik} - \sum_{i = 1}^{M} I \left( y_{i} = j \right) \cdot \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}} \cdot x_{ik} - \sum_{i = 1}^{M} \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}} \cdot x_{ik} \cdot \sum_{c \neq j}^{K} I \left( y_{i} = c \right) \right) \newline
  &= - \left( \sum_{i = 1}^{M} I \left( y_{i} = j \right) \cdot x_{ik} - \sum_{i = 1}^{M} I \left( y_{i} = j \right) \cdot \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}} \cdot x_{ik} - \sum_{i = 1}^{M} \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}} \cdot x_{ik} \cdot \left( 1 - I \left( y_{i} = j \right) \right) \right) \newline
  &= - \left( \sum_{i = 1}^{M} I \left( y_{i} = j \right) \cdot x_{ik} - \sum_{i = 1}^{M} \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}} \cdot x_{ik} \right) \newline
  &= - \sum_{i = 1}^{M} \left( x_{ik} \cdot \left( I \left( y_{i} = j \right) - \frac{\mathrm{e}^{\theta_{j}^{T} x_{i}}}{\sum_{s = 1}^{K} \mathrm{e}^{\theta_{s}^{T} x_{i}}} \right) \right) \newline
  \end{aligned}
  $$