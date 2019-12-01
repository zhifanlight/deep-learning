# 熵

## 熵（自信息）

- 描述一个随机变量所需要的平均信息量

- 假设 $p$ 是样本的真实分布，则熵计算如下：

  $$
  H \left( p \right) = -\sum_{i} p_{i} \cdot \log p_{i}
  $$

  - 底数是 $2$ 时，单位是比特

  - 底数是 $e$ 时，单位是奈特

- 熵的取值范围是 $\left[ 0, \ \log K \right]$，其中 $K$ 是样本的取值个数

  - 当只有一项 $p_{i}$ 取值为 $1$，而其他 $p_{i}$ 取值均为 $0$ 时，$H \left( p \right) = 0$

  - 当 $p_{i}$ 均匀分布，即 $p_{1} = p_{2} = \ \cdots \ = p_{K}$ 时，$H \left( p \right) = \log K$

    - 优化目标：

      $$
      f \left( p_{1}, \ p_{2}, \ \cdots, \ p_{K} \right) = -\sum_{k = 1}^{K} p_{k} \cdot \log p_{k}
      $$

    - 约束条件：

      $$
      \sum_{k=1}^{K} p_{k} = 1
      $$

    - 由拉格朗日乘子法：

      $$
      L \left( p, \ \lambda \right) = -\sum_{k = 1}^{K} p_{k} \cdot \log p_{k} + \lambda \left( \sum_{k = 1}^{K} p_{k} - 1 \right)
      $$

      - 计算偏导：

        $$
        \frac{\partial{L}}{\partial{p_{k}}} = - \left( \frac{1}{\ln 2} + \log p_{k} \right) - \lambda
        $$

        $$
        \frac{\partial{L}}{\partial{\lambda}} = \sum_{k = 1}^{K} p_{k} - 1
        $$

      - 令偏导为 $0$：

        $$
        p_{1} = p_{2} = \cdots = p_{k} = \frac{1}{K}
        $$

      - 代入信息熵计算公式可得：

        $$
        H \left( p \right) = \log K
        $$

## 相对熵（$\mathrm{KL}$ 散度）

- $\mathrm{Kullback-Leibler}$ 散度

- 衡量假设分布 $q$ 与真实分布 $p$ 之间的差距，计算如下：

  $$
  \mathrm{KL} \left( p||q \right) = \sum_{i} p_{i} \cdot \log \frac{p_{i}}{q_{i}}
  $$

- $\mathrm{KL}$ 散度不满足对称性：$\mathrm{KL} \left( p||q \right) \neq \mathrm{KL} \left( q||p \right)$

- $\mathrm{KL} \left( p||q \right) \geq 0$ 恒成立，当且仅当 $p = q$ 时等号成立。证明：

  $$
  \begin{aligned}
  \mathrm{prior} \qquad &\ln x \leq \left( x - 1 \right) \qquad \forall x \newline \newline 
  \mathrm{set} \qquad f &= - \mathrm{KL} \left( p||q \right) \newline
  &= \sum_{i} p_{i} \cdot \log \frac{q_{i}}{p_{i}} \newline
  &\leq \sum_{i} p_{i} \cdot \left( \frac{q_{i}}{p_{i}} - 1 \right) \newline
  &= \sum_{i} \left( q_{i} - p_{i} \right) \qquad \mathrm{iif} \ \ p_{i} = q_{i} \newline
  &= 0 \newline \newline
  \Rightarrow \qquad &\mathrm{KL} \left( p||q \right) \geq 0
  \end{aligned}
  $$

- 最小化 $\mathrm{KL}$ 散度等价于最大化对数似然。证明：

  - 用 $q \left( x; \ \theta \right)$ 来近似真实分布 $p \left( x \right)$，已知来自 $p \left( x \right)$ 的样本集 $S = \left\{ x_{1}, \ x_{2}, \ \cdots, \ x_{N} \right\}$：

    $$
    \begin{aligned}
    \min \mathrm{KL} \left( p||q \right) &= \sum_{i} p_{i} \cdot \log \frac{p_{i}}{q_{i}} \newline
    &= \sum_{i} p_{i} \cdot \left( \log p_{i} - \log q_{i} \right) \newline
    &= \sum_{x \in S} \frac{1}{N} \cdot \left( \log \frac{1}{N}  - \log q \left( x|\theta \right) \right) \newline \newline
    \min \mathrm{KL} \left( p||q \right) &= \max \sum_{x \in S} \log q \left( x|\theta \right)
    \end{aligned}
    $$

## 交叉熵

- 衡量假设分布 $q$ 与真实分布 $p$ 之间的差距，计算如下：

  $$
  H \left( p, \ q \right) = -\sum_{i} p_{i} \cdot \log q_{i}
  $$

- 在真实分布 $p$ 已知的情况下，交叉熵与相对熵在行为上等价：都反应真实分布 $p$ 和假设分布 $q$ 的差距

- 交叉熵是熵与相对熵之和：

  $$
  H \left( p, \ q \right) = H \left( p \right) + \mathrm{KL} \left( p||q \right)
  $$

### $\mathrm{Sigmoid}$ 交叉熵

$$
\begin{aligned}
H &= -y \cdot \log p - \left( 1 - y \right) \cdot \log \left( 1 - p \right) \newline
&= -y \cdot \log \left( \frac{1}{1 + \exp \left( -x \right)} \right) - \left( 1 - y \right) \cdot \log \left( 1 - \frac{1}{1 + \exp \left( -x \right)} \right) \newline
&= -y \cdot \log \left( \frac{1}{1 + \exp \left( -x \right)} \right) - \left( 1 - y \right) \cdot \log \left( \frac{\exp \left( -x \right)}{1 + \exp \left( -x \right)} \right) \newline
&= -y \cdot \log \left( \frac{1}{1 + \exp \left( -x \right)} \right) - \left( 1 - y \right) \cdot \left( -x + \log \left( \frac{1}{1 + \exp \left( -x \right)} \right) \right) \newline
&= \left( 1 - y \right) \cdot x - \log \left( \frac{1}{1 + \exp \left( -x \right)} \right) \newline
&= x - xy + \log \left( 1 + \exp \left( -x \right) \right) \newline
\end{aligned}
$$

- 当 $x < 0$ 时，为避免 $\exp \left( -x \right)$ 溢出，可对上式进行变换：

  $$
  \begin{aligned}
  H &= x - xy + \log \left( 1 + \exp \left( -x \right) \right) \newline
  &= -xy + \log \exp \left( x \right) + \log \left( 1 + \exp \left( -x \right) \right) \newline
  &= -xy + \log \left( \exp \left( x \right) + 1 \right) \newline
  \end{aligned}
  $$

- 综上所述，$\mathrm{Sigmoid}$ 交叉熵计算如下：

  $$
  H = \max \left( x, \ 0 \right) - xy + \log \left( 1 + \exp \left( -|x| \right) \right)
  $$

### $\mathrm{Softmax}$ 交叉熵

- 首先计算预测输出的 $\mathrm{Softmax}$ 值：

  $$
  p_{i} = \frac{\exp \left( x_{i} \right) }{\sum_{j} \exp \left( x_{j} \right)}
  $$

- 计算真实类别与预测输出的交叉熵：

  $$
  H = -\sum_{i} y_{i} \cdot \log p_{i}
  $$