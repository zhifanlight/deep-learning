# $\mathrm{kNN}$

## 基本思想

- 分类算法的一种，属于有监督学习

- 如果某样本的 $k$ 个最相似样本中的大多数都属于某个类别，那么该样本也很有可能属于该类别

## 算法流程

- 在特征空间中，寻找与当前样本最相似的 $k$ 个样本

- 将这 $k$ 个样本中出现次数最多的类别标签作为当前样本的类别标签

## 算法分析

- 没有明显的前期训练过程，将样本进行简单特征提取后，即可进行分类

- 对于给定实例 $x$，其最近邻的 $k$ 个样本集合为 $N \left( x \right)$，覆盖 $N \left( x \right)$ 的类别是 $c_{j}$，那么误分类率是：

  $$
  \begin{aligned}
  \epsilon &= \frac{1}{k} \sum_{x_{i} \in N \left( x \right)} I \left( y_{i} \neq c_{j} \right) \newline
  &= \frac{1}{k} \sum_{x_{i} \in N \left( x \right)} \left( 1 - I \left( y_{i} = c_{j} \right) \right) \newline
  &= 1 - \frac{1}{k} \sum_{x_{i} \in N \left( x \right)} I \left( y_{i} = c_{j} \right) \newline
  \end{aligned}
  $$

  - 多数表决等价于最大化 $\sum_{x_{i} \in N \left( x \right)}$，等价于最小化经验风险

## $\mathrm{KD}$ 树创建

- 从数据的第一维开始，循环重复以下过程：

  - 寻找当前维度的中位数，并将当前数据集分为两个子集

  - 递归的在下一个维度上处理两个子集

- 时间复杂度：

  $$
  T \left( n \right) = 2 \cdot T \left( \frac{n}{2} \right) + n \quad \Rightarrow \quad T \left( n \right) = O \left( n \log n \right)
  $$

## 维度灾难

- $\mathrm{kNN}$ 分类时，要计算当前样本到数据集中所有样本的距离，维度增加会导致大量计算

- 当维度增大时，待分类样本的单位距离内的其他样本数量会减少，需要寻找更远的距离才能找到 $k$ 个样本；但此时距离较远，找到的样本参考价值较小