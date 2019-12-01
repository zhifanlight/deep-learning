# 数据降维

## 背景介绍

- 机器学习中，数据被表示为向量；为了处理高维向量，需要消耗大量的计算资源

- 向量不同维度之间存在某种关系，增加了问题分析的复杂性，也导致了数据冗余

- 降维的目的就是在降低数据维度的同时，尽量减少关键信息的损失

## $\mathrm{PCA}$

- $\mathrm{Principal \ Component \ Analysis}$，即主成分分析

### 基本思想

- 假设原始空间维度为 $N$，降维后的空间维度为 $D$，其中 $D \leq N$

- 计算最佳投影矩阵，把 $N$ 维空间中的点投影到 $D$ 维空间，优化目标如下：

  - 同一维度的方差尽量大，保证投影后的点尽量分散

  - 不同维度的协方差尽量小，消除不同维度之间的依赖

### 推导过程

- 把 $N$ 维向量 $x_{i}$ 投影到 $D$ 维空间，相当于在 $N$ 维空间中选择 $D$ 个基向量，将 $x_{i}$ 在 $D$ 个基向量上的投影作为 $D$ 维空间中的坐标

- 矩阵化表示为：

  $$
  \left[ \begin{matrix} y_{1} & y_{2} & \cdots & y_{M} \end{matrix} \right] = \left[ \begin{matrix} p_{1} \\ p_{2} \\ \vdots \\ p_{D} \end{matrix} \right] \left[ \begin{matrix} x_{1} & x_{2} & \cdots & x_{M} \end{matrix} \right]
  $$

  - 其中 $M$ 是样本数，$y_{i}$ 是投影后的 $D$ 维列向量，$p_{i}$ 是 $N$ 维空间中投影基向量的行向量表示，$x_{i}$ 是投影前的 $N$ 维列向量

  - 上式简记为：

    $$
    Y = PX
    $$

- 假设 $X$ 的协方差矩阵为 $C$，以 $N = 2$ 为例：

  - $X$ 可表示为：

    $$
    X = \left[ \begin{matrix} a_{1} & a_{2} & \cdots & a_{M} \\ b_{1} & b_{2} & \cdots & b_{M} \end{matrix} \right]
    $$

  - 协方差矩阵 $C$ 计算如下：

    $$
    C = \frac{1}{M} \left[ \begin{matrix} \sum_{i = 1}^{M} \left( a_{i} - \mu_{a} \right)^{2} & \sum_{i = 1}^{M} \left( a_{i} - \mu_{a} \right) \cdot \left( b_{i} - \mu_{b} \right) \\ \sum_{i = 1}^{M} \left( b_{i} - \mu_{b} \right) \cdot \left( a_{i} - \mu_{a} \right) & \sum_{i = 1}^{M} \left( b_{i} - \mu_{b} \right)^{2} \end{matrix} \right]
    $$

  - 如果对 $X$ 提前进行中心化处理，即每一维度都减掉均值可得：

    $$
    C = \frac{1}{M} XX^{T}
    $$

- 而 $Y$ 的协方差矩阵 $D$ 计算如下：

  $$
  D = \frac{1}{M} YY^{T} = \frac{1}{M} \left( PX \right) \left( PX \right)^{T} = P \left( \frac{1}{M} XX^{T} \right) P^{T} = PCP^{T}
  $$

- 优化目标是最大化 $D$ 的对角线元素，并使非对角元素为 $0$

  - 由于 $C$ 是实对称矩阵，由实对称矩阵对角化可得：

    $$
    \Lambda = Q^{T}CQ
    $$

    - 其中 $\Lambda$ 为 $C$ 特征值组成的对角矩阵，$Q$ 的每一列是 $C$ 单位正交化的特征向量

    - 关于实对称矩阵对角化，参考 [$\mathrm{MatrixBasis.md}$](MatrixBasis.md)

  - 将 $C$ 的 $N$ 个特征值从大到小排列后，如果只保留 $Q$ 前 $D$ 列特征向量，此时 $\Lambda$ 为 $C$ 的前 $D$ 个特征值组成的对角矩阵

  - 由于特征值表示对应特征向量方向上特征的重要程度，令 $P$ 等于 $Q^{T}$ 的前 $D$ 行，即可满足 PCA 的优化目标

- 即取 $XX^{T}$ 的前 $D$ 个特征向量组成 $P$ 的每一行，对 $X$ 投影得到 $Y = PX$

### 计算过程

- 生成原始数据矩阵 $X$

- 对 $X$ 的每一维进行归一化处理

  - 首先进行中心化处理，减去每一维的均值；由推导过程决定

  - 然后进行方差单位化，除以每一维的标准差；消除不同维度间的数值影响

- 求解 $XX^{T}$ 的前 $D$ 个特征向量

  - 直接求解 $C = XX^{T}$ 的特征向量

    $$
    C x = \lambda x
    $$

    - 关于特征值求解，参考 [$\mathrm{MatrixBasis.md}$](MatrixBasis.md)

  - 对 $X$ 进行奇异值分解

    $$
    X = U \Sigma V^{T}
    $$

    - $U$ 的每一列都是 $XX^{T}$ 的特征向量

    - 关于奇异值分解，参考 [$\mathrm{MatrixDecomposition.md}$](MatrixDecomposition.md)

## $\mathrm{LDA}$

- $\mathrm{Linear \ Discriminant \ Analysis}$，即线性判别分析

### 基本思想

- 假设原始空间维度为 $N$，降维后的空间维度为 $D$，其中 $D \leq N$

- 计算投影矩阵 $W$，把 $N$ 维空间中样本投影到 $D$ 维空间，优化目标如下：

  - 让不同类尽量分开，即最大化不同类别间的距离

  - 让同一类尽量聚拢，即最小化同一类别内的距离

### 推导过程

- 以 $N$ 维空间投影到 $1$ 维空间的二分类为例：

  - $w$ 是 $N$ 维列向量

  - 投影点在同一直线上，结果 $y$ 为标量：

    $$
    y = w^{T}x
    $$

  - 设 $X_{0}, \ X_{1}$ 表示两类样本的集合，$\mu_{0}, \ \mu_{1}$ 表示两类样本在原始空间的均值向量，$\Sigma_{0}, \ \Sigma_{1}$ 表示两类样本在原始空间的协方差矩阵，$\sigma_{0}, \ \sigma_{1}$ 表示两类样本在投影空间的方差，计算如下：

  $$
  \sigma_{i} = \frac{1}{|X_{i}|} \sum_{x \in X_{i}} \left( w^{T}x - w^{T}\mu_{i} \right) ^ {2} = w^{T} \left( \frac{1}{|X_{i}|} \sum_{x \in X_{i}} \left( x - \mu_{i} \right) \left( x - \mu_{i} \right)^{T} \right) w = w^{T} \Sigma_{i} w
  $$

- 定义距离与散度矩阵如下：

  - 类间距离：

    $$
    ||w^{T}\mu_{0} - w^{T}\mu_{1}||^{2} = w^{T} \left( \mu_{0} - \mu_{1} \right) \left( \mu_{0} - \mu_{1} \right)^{T} w
    $$

  - 类内距离：

    $$
    \sigma_{0} + \sigma_{1} = w^{T} \left( \Sigma_{0} + \Sigma_{1} \right) w
    $$

  - 类间散度矩阵：

    $$
    S_{b} = \left( \mu_{0} - \mu_{1} \right) \left( \mu_{0} - \mu_{1} \right)^{T}
    $$

  - 类内散度矩阵：

    $$
    S_{w} = \Sigma_{0} + \Sigma_{1}
    $$

- 原问题相当于最大化：

  $$
  J \left( w \right) = \frac{w^{T} S_{b} w}{w^{T} S_{w} w}
  $$

  - 由于分子、分母都是 $w$ 的二次型，$J \left( w \right)$ 的解与 $w$ 长度无关，只与方向有关；由于 $w^{T} S_{w} w$ 为标量，对 $w$ 进行如下约束：

    $$
    w^{T} S_{w} w = 1
    $$

  - 上述问题转换为：

    $$
    \min -w^{T} S_{b} w \quad \mathrm{s.t.} \ w^{T} S_{w} w = 1
    $$

- 由拉格朗日乘子法：

  $$
   L \left( w, \lambda \right) = -w^{T} S_{b} w + \lambda \left( w^{T} S_{w} w - 1 \right)
  $$

  - 对 $w$ 求导并令导数为 $0$ 可得：

    $$
    S_{b} w = \lambda S_{w} w
    $$

- 由 $S_{b}$ 定义可知：

  $$
  S_{b} w = \left( \mu_{0} - \mu_{1} \right) \left( \left( \mu_{0} - \mu_{1} \right)^{T} w \right) = \left( \mu_{0} - \mu_{1} \right) \cdot \lambda_{w}
  $$

  - 代入上式可得：

    $$
    w = \frac{\lambda_{w}}{\lambda} S_{w}^{-1} \left( \mu_{0} - \mu_{1} \right)
    $$

  - 由于 $w$ 的长度不影响最终结果，进一步可得：

    $$
    w = S_{w}^{-1} \left( \mu_{0} - \mu_{1} \right)
    $$

### 计算过程

- 计算均值向量 $\mu_{0}, \ \mu_{1}$

- 计算协方差矩阵 $\Sigma_{0}, \ \Sigma_{1}$

- 计算类内散度矩阵的逆 $S_{w}^{-1} = \left( \Sigma_{0} + \Sigma_{1} \right)^{-1}$

- 计算投影向量 $w = S_{w}^{-1} \left( \mu_{0} - \mu_{1} \right)$

## $\mathrm{t-SNE}$

### 基本思想

- 假设原始空间维度为 $N$，降维后的空间维度为 $D$，其中 $D \leq N$

- $x_{1}, \ x_{2}, \ \cdots, \ x_{M}$ 是原始空间中的点，$y_{1}, \ y_{2}, \ \cdots, \ y_{M}$ 是投影空间中的点

- $x_{i}, \ x_{j}$ 的相似度矩阵为 $P$，$y_{i}, \ y_{j}$ 的相似度矩阵为 $Q$，$P, \ Q$ 的维度均为 $M \times M$

- $\mathrm{t-SNE}$ 的目标是最小化 $P, \ Q$ 间的 $\mathrm{KL}$ 散度；由于通过梯度法迭代，速度较慢

- 由于 $\mathrm{t-SNE}$ 是非线性降维，一般不用于后续的特征分析，主要作用是进行数据可视化