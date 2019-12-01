# 隐马尔可夫模型

## 数学定义

- 隐马尔可夫模型的形式定义如下：

  - 状态集合：

    $$
    Q = \{ q_{1}, \ q_{2}, \ \cdots , \ q_{N} \}
    $$

  - 观测集合：

    $$
    V = \{ v_{1}, \ v_{2}, \ \cdots , \ v_{M} \}
    $$

  - 状态序列：

    $$
    I = \left( i_{1}, \ i_{2}, \ \cdots , \ i_{T} \right)
    $$

  - 观测序列：

    $$
    O = \left( o_{1}, \ o_{2}, \ \cdots , \ o_{T} \right)
    $$

  - 状态转移矩阵：

    $$
    A = \left[ \ a_{ij} \ \right]_{N \times N}
    $$

    - 其中 $a_{ij}$ 表示由状态 $q_{i}$ 转移到状态 $q_{j}$ 的概率：

      $$
      a_{ij} = P \left( i_{t + 1} = q_{j}|i_{t} = q_{i} \right)
      $$

  - 观测概率矩阵：

    $$
    B = \left[ b_{j} \left( k \right) \right]_{N \times M}
    $$

    - 其中 $b_{j} \left( k \right)$ 表示状态 $q_{j}$ 下观测到 $v_{k}$ 的概率：

      $$
      b_{j} \left( k \right) = P \left( o_{t} = v_{k}|i_{t} = q_{j} \right)
      $$

  - 初始状态概率向量：

    $$
    \pi = \left( \pi_{i} \right)
    $$

    - 其中 $\pi_{i}$ 是初始时刻处于状态 $q_{i}$ 的概率：

      $$
      \pi_{i} = P \left( i_{1} = q_{i} \right)
      $$

- 隐马尔可夫模型三要素：

  - 状态转移概率矩阵 $A$

  - 观测概率矩阵 $B$

  - 初始状态概率向量 $\pi$

- 隐马尔可夫模型两个基本假设：

  - 任意时刻的状态只依赖于前一时刻的状态

  - 任意时刻的观测只依赖于该时刻的状态

- 隐马尔可夫模型三个基本问题：

  - 估值问题：

    - 给定模型和观测序列，计算观测序列出现的概率

  - 学习问题：

    - 给定观测序列，估计模型参数

  - 解码问题：

    - 给定模型和观测序列，求最有可能的状态序列

## 估值问题

### 前向算法

- 给定模型和观测序列，计算观测序列出现的概率

- 定义前向概率为 $t$ 时刻状态为 $q_{i}$、观测序列为 $o_{1}, \ o_{2}, \ \cdots , \ o_{t}$ 的概率：

  $$
  \alpha_{t} \left( i \right) = P \left( o_{1}, \ o_{2}, \ \cdots, \ o_{t}, \ i_{t} = q_{i} \right)
  $$

- 初始化：

  $$
  \alpha_{1} \left( i \right) = \pi_{i} b_{i} \left( o_{1} \right)
  $$

- 递推式：

  $$
  \alpha_{t + 1} \left( i \right) = \left[ \sum_{j = 1}^{N} \alpha_{t} \left( j \right) \cdot a_{ji} \right] \cdot b_{i} \left( o_{t + 1} \right)
  $$

- 终止：

  $$
  P \left( O|\lambda \right) = \sum_{i = 1}^{N} \alpha_{T} \left( i \right)
  $$

- 时间复杂度为 $O \left( N^{2}T \right)$：

  - 需要计算 $T$ 个时刻的概率

  - 每个时刻计算 $N$ 个概率

  - 每次计算需要前一时刻 $N$ 个状态的概率

### 后向算法

- 给定模型和观测序列，计算观测序列出现的概率

- 定义后向概率为 $t$ 时刻状态为 $q_{i}$ 的条件下、 从 $t + 1$ 到 $T$ 时刻观测序列为 $o_{t + 1}, \ o_{t + 2}, \ \cdots , \ o_{T}$ 的概率：

  $$
  \beta_{t} \left( i \right) = P \left( o_{t + 1}, \ o_{t + 2}, \ \cdots, \ o_{T} | i_{t} = q_{i} \right)
  $$

- 初始化：

  $$
  \beta_{T} \left( i \right) = 1
  $$

- 递推式：

  $$
  \beta_{t} \left( i \right) = \sum_{j = 1}^{N} \beta_{t + 1} \left( j \right) \cdot a_{ij} \cdot b_{j} \left( o_{t + 1} \right)
  $$

- 终止：

  $$
  P \left( O|\lambda \right) = \sum_{i = 1}^{N} \beta_{1} \left( i \right) \cdot \pi_{i} \cdot b_{i} \left( o_{1} \right)
  $$

- 时间复杂度为 $O \left( N^{2}T \right)$：

  - 需要计算 $T$ 个时刻的概率

  - 每个时刻计算 $N$ 个概率

  - 每次计算需要后一时刻 $N$ 个状态的概率

## 解码问题

### 维特比算法

- 给定模型和观测序列，求最有可能的状态序列

- 在 $t$ 时刻状态为 $i$ 的所有路径 $i_{1}, \ i_{2}, \ \cdots, \ i_{t}$ 中概率最大值为：

  $$
  \delta_{t} \left( i \right) = \max P \left( i_{t} = i, \ i_{t - 1}, \cdots, i_{1}, \ o_{t}, \ \cdots, o_{1} \right)
  $$

  - 产生当前最大概率值的结点为 $j$

- 初始化：

  $$
  \delta_{1} \left( i \right) = \pi_{i} b_{1} \left( i \right)
  $$

  $$
  \Phi_{1} \left( i \right) = 0
  $$

- 递推式：

  $$
  \delta_{t} \left( i \right) = \max \left[ \delta_{t - 1} \left( j \right) \cdot a_{ji} \right] \cdot b_{i} \left( o_{t} \right)
  $$

  $$
  \Phi_{t} \left( i \right) = \arg \max_{j} \left[ \delta_{t - 1} \left( j \right) \cdot a_{ji} \right]
  $$

- 终止：

  $$
  P^{*} = \max \delta_{T} \left( i \right)
  $$

  $$
  i_{T}^{*} = \arg \max_{i} \delta_{T} \left( i \right)
  $$

- 回溯：

  $$
  i_{t}^{*} = \Phi_{t + 1} \left( i_{t + 1}^{*} \right)
  $$