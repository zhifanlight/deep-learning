# 距离度量

## 闵可夫斯基距离

- $N$ 维空间中，两个样本 $x, \ y$ 间的闵可夫距离定义为：

  $$
  D \left( x, \ y \right) = \left( \sum_{i = 1}^{N} |x_{i} - y_{i}|^{p} \right) ^{\frac{1}{p}}
  $$

- $p = 1$ 时，得到曼哈顿距离：

  $$
  D \left( x, \ y \right) = \sum_{i = 1}^{N} |x_{i} - y_{i}|
  $$

- $p = 2$ 时，得到欧几里德距离：

  $$
  D \left( x, \ y \right) = \sqrt{\sum_{i = 1}^{N} \left( x_{i} - y_{i} \right)^{2}}
  $$

- $p \rightarrow \infty$ 时，得到切比雪夫距离：

  $$
  D \left( x, \ y \right) = \max_{i = 1}^{N} |x_{i} - y_{i}|
  $$

## 交叉熵

- 参考 [$\mathrm{Entropy.md}$](Entropy.md)

## $\mathrm{KL}$ 散度

- $\mathrm{Kullback-Leibler}$ 散度

- 参考 [$\mathrm{Entropy.md}$](Entropy.md)

## $\mathrm{JS}$ 散度

- $\mathrm{Jenssen-Shannon}$ 散度

- $\mathrm{JS} \left( p, \ q \right) = \frac{1}{2} \mathrm{KL} \left( p||\frac{p + q}{2} \right) + \frac{1}{2} \mathrm{KL} \left( q||\frac{p + q}{2} \right)$，其中 $p$ 是样本的真实分布，$q$ 是样本的假设分布

- 与 $\mathrm{KL}$ 相比，$\mathrm{JS}$ 散度具有对称性：$\mathrm{JS} \left( p, \ q \right) = \mathrm{JS} \left( q, \ p \right)$

- 当真实分布 $p$ 与假设分布 $q$ 不重叠时，$\mathrm{JS}$ 散度为常数 $\log 2$：

  $$
  \begin{aligned}
  \mathrm{prior} \quad &p_{i} = 0 \quad \mathrm{when} \quad q_{i} \geq 0, \qquad q_{i} = 0 \quad \mathrm{when} \quad p_{i} \geq 0 \newline \newline
  \qquad \mathrm{JS} \left( p, \ q \right) &= \frac{1}{2} \sum_{i} p_{i} \log \frac{2 p_{i}}{p_{i} + q_{i}} + \frac{1}{2} \sum_{i} q_{i} \log \frac{2 q_{i}}{p_{i} + q_{i}} \newline
  &= \frac{1}{2} \log 2 \ \sum_{i} p_{i} + \frac{1}{2} \sum_{i} p_{i} \log \frac{p_{i}}{p_{i} + q_{i}} + \frac{1}{2} \log 2 \ \sum_{i} q_{i} + \frac{1}{2} \sum_{i} q_{i} \log \frac{q_{i}}{p_{i} + q_{i}} \newline
  &= \log 2 + \frac{1}{2} \sum_{i} p_{i} \log \frac{p_{i}}{p_{i} + q_{i}} + \frac{1}{2} \sum_{i} q_{i} \log \frac{q_{i}}{p_{i} + q_{i}} \qquad (1) \newline
  &= \log 2 + \frac{1}{2} \sum_{p_{i} \neq 0} p_{i} \log \frac{p_{i}}{p_{i}} + \frac{1}{2} \sum_{q_{i} \neq 0} q_{i} \log \frac{q_{i}}{q_{i}} \qquad (2) \newline
  &= \log 2
  \end{aligned}
  $$

- 由 $\left( 1 \right)$ 到 $\left( 2 \right)$ 的推导，由洛必达法则可得：

  $$
  \left\{ \begin{matrix} p_{i} \log \frac{p_{i}}{p_{i} + q_{i}} = 0 & \mathrm{if} \ p_{i} = 0, \ q_{i} \neq 0 \\ q_{i} \log \frac{q_{i}}{p_{i} + q_{i}} = 0 & \mathrm{if} \ q_{i} = 0, \ p_{i} \neq 0 \end{matrix} \right.
  $$

## $\mathrm{Wasserstein}$ 距离

- 又称 $\mathrm{EM}$ 距离，即 $\mathrm{Earth-Mover}$ 距离

- $W \left( p, \ q \right) = \inf \limits_{\gamma \sim \Pi \left( p, \ q \right)} \mathbb{E}_{\left( x, \ y \right) \sim \gamma}{\left[||x - y||\right]}$，其中 $\Pi \left( p, \ q \right)$ 是 $p$ 和 $q$ 所有可能的联合分布的集合，$\left( x, \ y \right)$ 是从联合分布 $\gamma$ 中采样的样本，$inf$ 表示最大下界

- $\mathbb{E}_{\left( x, \ y \right) \sim \gamma}{\left[||x - y||\right]}$ 可以理解为在 $\gamma$ 这个“路径规划”下，把 $q$ 这堆“土”挪到 $p$ “位置”所需的“消耗”，而 $W \left( p, \ q \right)$ 就是在“最优路径规划”下的的“最小消耗”

- 与 $\mathrm{JS}$ 散度相比，即使没有重叠，$\mathrm{Wasserstein}$ 距离也能够反映两个分布的距离：

  - 考虑二维平面内两个均匀分布 $P_{1}, \ P_{2}$:

  <center>
  <img src="images/wasserstein.png"/>
  </center>

    - $JS \left( P_{1}||P_{2} \right) = \left\{ \begin{matrix} \log 2 & \mathrm{if} \ \theta \neq 0 \\ 0 & \mathrm{if} \ \theta = 0 \end{matrix} \right.$

    - $W \left( P_{1}, \ P_{2} \right) = |\theta|$