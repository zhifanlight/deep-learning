# 支持向量机

## 背景介绍

- 对于给定数据集的分类任务，判别面通常不止一个

- $\mathrm{SVM}$ 通过最大化正、负支持向量的间隔，来寻找最优判别面，以提高泛化性能

## 数学推导

### 线性可分

- 假设样本集线性可分，判别面为 $h \left( x \right) = w^{T} x + b$，类别标签 $y_{i} \in \left\{ -1, \ +1 \right\}$，规则如下：

  $$
  \left\{ \begin{matrix} y_{i} = +1 \quad \Leftrightarrow \quad w^{T} x_{i} + b > 0 \\ y_{i} = -1 \quad \Leftrightarrow \quad w^{T} x_{i} + b < 0  \end{matrix} \right.
  $$

- 统一为 $y_{i} \cdot \left( w^{T} x_{i} + b \right) > 0$，通过对 $w$ 缩放，可进一步保证 $y_{i} \cdot \left( w^{T} x_{i} + b \right) \geq 1$，而满足 $y_{i} \cdot \left( w^{T} x_{i} + b \right) = 1$ 的样本称为支持向量

- $\mathrm{SVM}$ 优化目标是最大化正、负支持向量间隔：

  $$
  \max_{w, \ b} 2 \frac{|w^{T}x_{i} + b|}{||w||} = \max_{w, \ b} \frac{2}{||w||}
  $$

- 由于上述目标函数求解困难，可以进行如下转换：

  $$
  \max_{w, \ b} \frac{2}{||w||} \quad \Leftrightarrow \quad \min_{w, \ b} \frac{1}{2} ||w||^{2}
  $$

  - 约束条件为：

    $$
    y_{i} \cdot \left( w^{T} x_{i} + b \right) \geq 1, \quad i = 1, \ 2, \ \cdots, \ M
    $$

- 由拉格朗日乘子法可得：

  $$
  L \left( w, \ b, \ \alpha \right) = \frac{1}{2} ||w||^{2} + \sum_{i = 1}^{M} \alpha_{i} \cdot \left( 1 - y_{i} \cdot \left( w^{T} x_{i} + b \right) \right)
  $$

  - $\mathrm{KKT}$ 条件为：

    $$
    \alpha_{i} \geq 0, \quad i = 1, \ 2, \ \cdots, \ M
    $$

    $$
    \alpha_{i} \cdot \left( 1 - y_{i} \cdot \left( w^{T} x_{i} + b \right) \right) = 0, \quad i = 1, \ 2, \ \cdots, \ M
    $$

- 计算 $\frac{\partial L}{\partial w}, \ \frac{\partial L}{\partial b}$ 并令偏导数为 $0$ 可得：

  $$
  w = \sum_{i = 1}^{M} \alpha_{i} y_{i} x_{i} \qquad \sum_{i = 1}^{M} \alpha_{i} y_{i} = 0
  $$

  - 将上述结果代入 $L \left( w, \ b, \ \alpha \right)$ 可得：

    $$
    L \left( w, \ b, \ \alpha \right) = \sum_{i = 1}^{M} \alpha_{i} - \frac{1}{2} \sum_{i = 1}^{M} \sum_{j = 1}^{M} \alpha_{i} \alpha_{j} \cdot y_{i} y_{j} \cdot x_{i}^{T} x_{j}
    $$

- 根据拉格朗日乘子法对偶性，上述问题可转化为：

  $$
  \max_{\alpha} G \left( \alpha \right) = \max_{\alpha} \left( \sum_{i = 1}^{M} \alpha_{i} - \frac{1}{2} \sum_{i = 1}^{M} \sum_{j = 1}^{M} \alpha_{i} \alpha_{j} \cdot y_{i} y_{j} \cdot x_{i}^{T} x_{j} \right)
  $$

  - 约束条件为：

    $$
    \alpha_{i} \geq 0, \quad i = 1, \ 2, \ \cdots, \ M
    $$

    $$
    \sum_{i = 1}^{M} \alpha_{i} y_{i} = 0, \quad i = 1, \ 2, \ \cdots, \ M
    $$

### 线性不可分

- 在判别面附近，如果完全按照原始约束条件，会导致 $\mathrm{SVM}$ 产生较大的误差

- 为让 $\mathrm{SVM}$ 忽略某些噪声，可以引入松弛变量 $\xi_{i} \geq 0$ 来允许错误分类发生：

  $$
  y_{i} \cdot \left( w^{T} x_{i} + b \right) \geq 1 - \xi_{i}, \quad i = 1, \ 2, \ \cdots, \ M
  $$

  - 对 $\xi_{i}$ 添加约束如下：

    $$
    \min_{\xi} C \sum_{i = 1}^{M} \xi_{i}
    $$

- 此时 $\mathrm{SVM}$ 的优化目标变为：

  $$
  \min_{w, \ b, \ \xi} \frac{1}{2} ||w||^{2} + C \sum_{i = 1}^{M} \xi_{i}
  $$

  - $C$ 是惩罚因子，$C$ 越大，表明越不希望离群点出现

  - 约束条件为：

    $$
    \xi_{i} \geq 0, \quad i = 1, \ 2, \ \cdots, \ M
    $$

    $$
    y_{i} \cdot \left( w^{T} x_{i} + b \right) \geq 1 - \xi_{i}, \quad i = 1, \ 2, \ \cdots, \ M
    $$

- 由拉格朗日乘子法可得：

  $$
  L \left( w, \ b, \ \alpha \right) = \frac{1}{2} ||w||^{2} + C \sum_{i = 1}^{M} \xi_{i} + \sum_{i = 1}^{M} \alpha_{i} \cdot \left( 1 - \xi_{i} - y_{i} \cdot \left( w^{T} x_{i} + b \right) \right) - \sum_{i = 1}^{M} \beta_{i} \cdot \xi_{i}
  $$

  - $\mathrm{KKT}$ 条件为：

    $$
    \alpha_{i} \geq 0, \quad i = 1, \ 2, \ \cdots, \ M
    $$

    $$
    \beta_{i} \geq 0, \quad i = 1, \ 2, \ \cdots, \ M
    $$

    $$
    \beta_{i} \xi_{i} = 0, \quad i = 1, \ 2, \ \cdots, \ M
    $$

    $$
    \alpha_{i} \cdot \left( 1 - \xi_{i} - y_{i} \cdot \left( w^{T} x_{i} + b \right) \right) = 0, \quad i = 1, \ 2, \ \cdots, \ M
    $$

- 计算 $\frac{\partial L}{\partial w}, \ \frac{\partial L}{\partial b}, \ \frac{\partial L}{\partial \xi_{i}}$ 并令偏导数为 $0$ 可得：

  $$
  w = \sum_{i = 1}^{M} \alpha_{i} y_{i} x_{i} \qquad \sum_{i = 1}^{M} \alpha_{i} y_{i} = 0 \qquad \alpha_{i} + \beta_{i} = C
  $$

  - 将上述结果代入 $L \left( w, \ b, \ \alpha \right)$ 可得：

    $$
    L \left( w, \ b, \ \alpha \right) = \sum_{i = 1}^{M} \alpha_{i} - \frac{1}{2} \sum_{i = 1}^{M} \sum_{j = 1}^{M} \alpha_{i} \alpha_{j} \cdot y_{i} y_{j} \cdot x_{i}^{T} x_{j} 
    $$

- 根据拉格朗日乘子法对偶性，上述问题可转化为：

  $$
  \max_{\alpha} G \left( \alpha \right) = \max_{\alpha} \left( \sum_{i = 1}^{M} \alpha_{i} - \frac{1}{2} \sum_{i = 1}^{M} \sum_{j = 1}^{M} \alpha_{i} \alpha_{j} \cdot y_{i} y_{j} \cdot x_{i}^{T} x_{j} \right)
  $$

  - 约束条件为：

    $$
    0 \leq \alpha_{i} \leq C, \quad i = 1, \ 2, \ \cdots, \ M
    $$

    $$
    \sum_{i=1}^{M} \alpha_{i} y_{i} = 0, \quad i = 1, \ 2, \ \cdots, \ M
    $$

### 优化求解

#### 求解 $\alpha$

- 使用 $\mathrm{SMO}$ 算法求解

- 优化目标：$G \left( \alpha \right)$，每次固定其他变量，只更新 $\alpha_{i}, \ \alpha_{j}$，直到收敛

- 由 $\sum \alpha_{i} y_{i} = 0$ 可得：

  $$
  \alpha_{i} y_{i} + \alpha_{i} y_{j} = -\sum_{k \neq i \neq j}^{M} \alpha_{k} y_{k} = D
  $$

  - 由 $y_{i} \in \left\{ -1, \ +1 \right\}$ 可得 $y_{i} = \frac{1}{y_{i}}$，上式可进一步变为：

    $$
    \alpha_{j} = \left( D - \alpha_{i} y_{i} \right) \cdot y_{j}
    $$

- 固定其他变量，将 $\alpha_{j}$ 代入 $G \left( \alpha \right)$ 可得：

  $$
  G \left( \alpha \right) = A \alpha_{i}^{2} + B \alpha_{i} + E
  $$

- 由边界条件 $0 \leq \alpha_{i}, \ \alpha_{j} \leq C$ 可知，该问题为区间内的二次函数最值问题，容易求解

#### 求解 $w$

- 求解出最优判别面的 $\alpha_{i}$ 后，代入下式计算 $w$：

  $$
  w = \sum_{i = 1}^{M} \alpha_{i} y_{i} x_{i}
  $$

#### 求解 $b$

- 假设 $V$ 为支持向量的集合，对于支持向量 $x_{v}$：

  $$
  y_{v} \left( \sum_{i = 1}^{M} \alpha_{i} y_{i} x_{i}^{T} x_{v} + b \right) = 1
  $$

- 由上式可得：

  $$
  b = \frac{1}{y_{v}} - \sum_{i = 1}^{M} \alpha_{i} y_{i} x_{i}^{T} x_{v} = y_{v} - \sum_{i = 1}^{M} \alpha_{i} y_{i} x_{i}^{T} x_{v}
  $$

- 通常在整个支持向量集合上求解 $b$：

  $$
  b = \frac{1}{|V|} \sum_{v \in V} \left( y_{v} - \sum_{i = 1}^{M} \alpha_{i} y_{i} x_{i}^{T} x_{v} \right)
  $$

### 支持向量

- 将 $w, \ b$ 代入 $h \left( x \right)$ 可得最终判别面：

  $$
  h \left( x \right) = \sum_{i = 1}^{M} \alpha_{i} y_{i} x_{i}^{T} x + \frac{1}{|V|} \sum_{v \in V} \left( y_{v} - \sum_{i = 1}^{M} \alpha_{i} y_{i} x_{i}^{T} x_{v} \right)
  $$

#### 线性可分

- 由约束条件 $\alpha_{i} \cdot \left( 1 - y_{i} \cdot \left( w^{T} x_{i} + b \right) \right) = 0$ 和 $\alpha_{i} \geq 0$ 可知：

  - 当 $\alpha_{i} = 0$ 时，样本 $\left( x_{i}, \ y_{i} \right)$ 对 $h \left( x \right)$ 无影响

  - 当 $\alpha_{i} > 0$ 时，$y_{i} \cdot \left( w^{T} x_{i} + b \right) = 1$，$\left( x_{i}, \ y_{i} \right)$ 为支持向量

#### 线性不可分

- 由约束条件 $\alpha_{i} \cdot \left( 1 - \xi_{i} - y_{i} \cdot \left( w^{T} x_{i} + b \right) \right) = 0$ 和 $0 \leq \alpha_{i} \leq C$ 可知：

  - 当 $\alpha_{i} = 0$ 时，样本 $\left( x_{i}, \ y_{i} \right)$ 对 $h \left( x \right)$ 无影响

  - 当 $0 < \alpha_{i} \leq C$ 时，$y_{i} \cdot \left( w^{T} x_{i} + b \right) = 1 - \xi_{i}$，$\left( x_{i}, \ y_{i} \right)$ 为支持向量：

- 当 $0 < \alpha_{i} \leq C$ 时，由约束条件 $\beta_{i} \cdot \xi_{i} = 0$ 和 $ 0 \leq \beta_{i} \leq C$ 可知：

  - 当 $0 < \alpha_{i} < C $ 时， $\beta_{i} > 0 \ \rightarrow \ \xi_{i} = 0$，样本 $\left( x_{i}, \ y_{i} \right)$ 在最大间隔边界上

  - 当 $\alpha_{i} = C $ 时，$\beta_{i} = 0 \ \rightarrow \ \xi_{i} > 0$：

    - 若 $\xi_{i} \leq 1$，样本 $\left( x_{i}, \ y_{i} \right)$ 在最大间隔内部

    - 若 $\xi_{i} > 1$，样本 $\left( x_{i}, \ y_{i} \right)$ 分类错误

#### 判别面

- 综上所述，判别面只与支持向量有关，最终表达式为：

  $$
  h \left( x \right) = \sum_{v \in V} \alpha_{v} y_{v} x_{v}^{T} x + \frac{1}{|V|} \sum_{v \in V} \left( y_{v} - \sum_{s \in V} \alpha_{s} y_{s} x_{s}^{T} x_{v} \right)
  $$