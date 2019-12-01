# 提升树

## 背景介绍

- 以分类树或回归树为基本分类器的提升方法

- 已知训练集 $\left\{ \left( x_{1}, \ y_{1} \right), \ \left( x_{2}, \ y_{2} \right), \ \cdots, \ \left( x_{M}, \ y_{M} \right) \right\}$

## 提升树

### 二叉分类树

- 二分类问题使用指数损失函数，只需将 $\mathrm{Adaboost}$ 中的分类器限制为二叉决策树即可

- 参考 [$\mathrm{Adaboost.md}$](Adaboost.md)

### 二叉回归树

- 回归问题使用平方损失函数，使用 $\mathrm{CART}$ 回归树

- 如果将输入空间划分为 $J$ 个不相交区域 $R_{1}, \ R_{2}, \ \cdots, \ R_{J}$，且在每个区域上的输出为常量 $c_{j}$，由此得到的提升树可表示为

  $$
  T \left( x; \ \theta \right) = \sum_{j = 1}^{J} c_{j} \cdot I \ \left( x \in R_{j} \right)
  $$

#### 数学推导

- 在算法的第 $t$ 步，给定当前模型 $f_{t - 1} \left( x \right)$，需要求解：

  $$
  \theta_{t}^{*} = \arg \min_{\theta} \sum_{i = 1}^{M} L \left( y_{i}, \ f_{t - 1} \left( x_{i} \right) + T \left( x_{i}; \ \theta_{t} \right) \right)
  $$

- 对于平方损失函数：

  $$
  L \left( y_{i}, \ f_{t - 1} \left( x_{i} \right) + T \left( x_{i}; \ \theta \right) \right) = \left[ y_{i} - f_{t - 1} \left( x_{i} \right) - T \left( x_{i}; \ \theta \right) \right]^{2} = \left[ r_{t, \ i} - T \left( x_{i}; \ \theta \right) \right]^{2}
  $$

  - 其中 $r_{t, \ i}$ 表示本轮（第 $t$ 轮）开始时 $x_{i}$ 的残差

  - 因此当前模型是在拟合上一轮的残差

#### 算法流程

- 初始化：

  $$
  f_{0} \left( x \right) = 0
  $$

- 对于第 $1, \ 2, \cdots, \ T$ 轮：

  - 计算每一个样本的残差：

    $$
    r_{t, \ i} = y_{i} - f_{t - 1} \left( x_{i} \right)
    $$

  - 拟合残差 $r_{t, \ i}$ 得到一棵回归树：

    $$
    T \left( x; \ \theta_{t} \right) = \sum_{j = 1}^{J} c_{j} \cdot I \ \left( x \in R_{j} \right)
    $$

  - 更新回归树：

    $$
    f_{t} \left( x \right) = f_{t - 1} \left( x \right) + T \left( x; \ \theta_{t} \right)
    $$

- 得到最终的提升树：

  $$
  f_{T} \left( x \right) = \sum_{t = 1}^{T} T \left( x; \ \theta_{t} \right)
  $$

## $\mathrm{GBDT}$

- $\mathrm{Gradient \ Boosting \ Decision \ Tree}$，即梯度提升决策树

- 当使用指数损失函数（分类）和平方损失函数（回归）时，提升树的优化都比较简单；但对于一般的损失函数，不太容易优化

- $\mathrm{GBDT}$ 利用损失函数在当前模型的负梯度（响应值）来代替残差：

  $$
  r_{t, \ i} = - \left[ \frac{\partial{L \left( y_{i}, \ f_{t - 1} \left( x_{i} \right) \right)}}{\partial{f_{t - 1} \left( x_{i} \right)}} \right]
  $$

  - 当使用平方损失函数时，$r_{t, \ i}$ 即为实际的残差

### 算法流程

- 初始化：

  $$
  f_{0} \left( x \right) = \arg \min_{c} \sum_{i = 1}^{M} L \left( y_{i}, \ c \right)
  $$

- 对于第 $1, \ 2, \cdots, \ T$ 轮：

  - 计算每一个样本的响应值：

    $$
    r_{t, \ i} = - \left[ \frac{\partial{L \left( y_{i}, \ f_{t - 1} \left( x_{i} \right) \right)}}{\partial{f_{t - 1} \left( x_{i} \right)}} \right]
    $$

  - 拟合响应值 $r_{t, \ i}$ 得到一棵回归树：

    $$
    T \left( x; \ \theta_{t} \right) = \sum_{j = 1}^{J} c_{j} \cdot I \ \left( x \in R_{j} \right)
    $$

  - 更新回归树：

    $$
    f_{t} \left( x \right) = f_{t - 1} \left( x \right) + T \left( x; \ \theta_{t} \right)
    $$

- 得到最终的 $\mathrm{GBDT}$：

  $$
  f_{T} \left( x \right) = \sum_{t = 1}^{T} T \left( x; \ \theta_{t} \right)
  $$

## $\mathrm{XGBoost}$

- 相对 $\mathrm{GBDT}$，基分类器除了回归树，还支持线性分类器

  - 分类问题：带 $L_{1}$ 和 $L_{2}$ 正则项的逻辑回归

  - 回归问题：带 $L_{1}$ 和 $L_{2}$ 正则项的线性回归

- 显式将树的复杂度作为正则项加入优化目标：

  $$
  \Omega \left( f_{t} \right) = \frac{\gamma}{2} \ J + \frac{\lambda}{2} \sum_{j = 1}^{J} b_{j}^{2}
  $$

  - 其中 $J$ 为回归树中叶结点的数量，$b_{j}$ 表示第 $j$ 个叶结点的输出

- 权重衰减：在进行完一次迭代后，将叶结点的权值乘上衰减系数，以削弱每棵树的影响，让后面有更大的学习空间

- 支持自定义损失函数，优化过程同时用到一阶导数和二阶导数：

  $$
  L_{t} \approx \sum_{i = 1}^{M} \left( L \left( y_{i}, \ f_{t - 1} \left( x_{i} \right) \right) + g_{i} \cdot T \left( x; \ \theta \right) + \frac{1}{2} h_{i} \cdot T^{2} \left( x; \ \theta \right) \right) + \Omega \left( f_{t} \right)
  $$

  - 其中，$g_{i}, \ h_{i}$ 分别为损失函数对当前模型的一阶导数、二阶导数：

    $$
    g_{i} = \frac{\partial{L \left( y_{i}, \ f_{t - 1} \left( x \right) \right)}}{\partial{f_{t - 1}}} \qquad g_{i} = \frac{\partial^{2}{L \left( y_{i}, \ f_{t - 1} \left( x \right) \right)}}{\partial^{2}{f_{t - 1}}}
    $$