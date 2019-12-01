# 损失函数

## 背景介绍

- 非负实值函数，用来估计模型的预测值 $f \left( x \right)$ 对真实值 $y$ 的拟合程度；损失函数越小，模型越好；通常用 $L \left( y, \ f \left( x \right) \right)$ 表示

- 一般由经验风险项、正则化项组成，定义如下：

  $$
  J \left( \theta \right) = \frac{1}{N} \sum_{i = 1}^{N} L \left( y_{i}, \ f \left( x_{i}; \ \theta \right) \right) + \lambda \cdot R \left( \theta \right)
  $$

- 关于正则化，参考 [$\mathrm{Regularization.md}$](Regularization.md)

## 常用损失函数

### $\mathrm{0-1}$ 损失函数

- 对于分类标签 $f \left( x_{i} \right)$ 与真实标签 $y_{i}$，相同时为 $0$，不同时为 $1$：

  $$
  L \left( y_{i}, \ f \left( x_{i} \right) \right) = \left\{ \begin{matrix} 1, \quad f \left( x_{i} \right) \neq y_{i} \\ 0, \quad f \left( x_{i} \right) = y_{i} \end{matrix} \right.
  $$

- 不易求导，因此在模型训练时不常用；通常用于模型的评价

### 平方损失函数

- 将预测值 $f \left( x_{i} \right)$ 与真实值 $y_{i}$ 之间的欧式距离作为误差：

  $$
  L \left( y_{i}, \ f \left( x_{i} \right) \right) = \left( y_{i} - f \left( x_{i} \right) \right)^{2}
  $$

  - 用于线性回归，参考 [$\mathrm{LinearRegression.md}$](LinearRegression.md)

### 对数损失函数

- 来源于最大似然估计：最大化对数似然等于最小化损失函数

- 将预测类别 $f \left( x_{i} \right)$ 与真实类别 $y_{i}$ 之间的交叉熵作为误差：

  - 二分类：

    $$
    p_{i} = \mathrm{sigmoid} \left( f \left( x_{i} \right) \right)
    $$

    $$
    L \left( y_{i}, \ f \left( x_{i} \right) \right) = - \left( y_{i} \cdot \log p_{i} + \left( 1 - y_{i} \right) \cdot \log \left( 1 - p_{i} \right) \right)
    $$

    - 用于逻辑回归，参考 [$\mathrm{LogisticRegression.md}$](LogisticRegression.md)

  - 多分类：

    $$
    p_{i} = \mathrm{softmax} \left( f \left( x_{i} \right) \right)
    $$

    $$
    L \left( y_{i}, \ f \left( x_{i} \right) \right) = - \sum_{j} y_{ij} \cdot \log p_{ij}
    $$

    - 用于 $\mathrm{Softmax}$ 回归，参考 [$\mathrm{SoftmaxRegression.md}$](SoftmaxRegression.md)

### 指数损失函数

- 标准形式如下：

  $$
  L \left( y_{i}, \ f \left( x_{i} \right) \right) = e^{-y_{i} \cdot f \left( x_{i} \right)}
  $$

  - 用于 $\mathrm{Adaboost}$，参考 [$\mathrm{Adaboost.md}$](Adaboost.md)

### $\mathrm{Hinge}$ 损失函数

- 标准形式如下：

  $$
  L \left( y_{i}, \ f \left( x_{i} \right) \right) = \max \left( 0, \ 1 - y_{i} \cdot f \left( x_{i} \right) \right)
  $$

  - 用于软间隔 $\mathrm{SVM}$，参考 [$\mathrm{SVM.md}$](SVM.md)

    - 由 $\xi_{i}$ 约束条件可得：

      $$
      \xi_{i} \geq \max \left( 0, \ 1 - y_{i} \cdot h \left( x_{i} \right) \right)
      $$

    - 代入目标函数可得：

      $$
      \min_{w, \ b, \ \xi} \frac{1}{2} ||w||^{2} + C \sum_{i = 1}^{M} \max \left( 0, \ 1 - y_{i} \cdot h \left( x_{i} \right) \right)
      $$

    - 因此软间隔 $\mathrm{SVM}$ 的损失函数为 $\mathrm{Hinge}$ 损失与 $L_{2}$ 正则项之和