# $\mathrm{EM}$ 算法

## 基本思想

- 给定 $m$ 个样本 $\left\{ x_{1}, \ x_{2}, \ \cdots, \ x_{m} \right\}$，假设样本之间相互独立

- 想要拟合模型 $P \left( x, \ z \right)$ 的参数 $\theta$，但只有 $x$ 可见，而隐变量 $z$ 不可见

- 交叉进行 $E$ 步和 $M$ 步：

  - $E$ 步：计算隐变量 $z$ 的后验分布 $P \left( z_{i}|x_{i} \right)$

  - $M$ 步：根据当前隐变量后验概率，更新模型参数

- 由后续推导可知，$M$ 步通过最大化对数似然的下界，来逼近对数似然函数

## 数学推导

- 对数似然函数如下：

  $$
  L \left( \theta \right) = \sum_{i = 1}^{m} \log P \left( x_{i}; \ \theta \right) = \sum_{i = 1}^{m} \log \left( \sum_{z_{i}} P \left( x_{i}, \ z_{i}; \ \theta \right) \right)
  $$
  
- 假设 $Q_{i} \left( z_{i} \right)$ 是概率分布，$L \left( \theta \right)$ 可推导为：

  $$
  L \left( \theta \right) = \sum_{i = 1}^{m} \log \left( \sum_{z_{i}} Q_{i} \left( z_{i} \right) \cdot \frac{P \left( x_{i}, \ z_{i}; \ \theta \right)}{Q_{i} \left( z_{i} \right)} \right) = \sum_{i = 1}^{m} \log \mathbb{E} \left[ \frac{P \left( x_{i}, \ z_{i}; \ \theta \right)}{Q_{i} \left( z_{i} \right)} \right]
  $$

- 由于对数函数 $\log$ 为凹函数，由 $\mathrm{Jensen}$ 不等式：

  $$
  L \left( \theta \right) \geq \sum_{i = 1}^{m} \mathbb{E} \left[ \log \left( \frac{P \left( x_{i}, \ z_{i}; \ \theta \right)}{Q_{i} \left( z_{i} \right)} \right) \right] = \sum_{i = 1}^{m} \sum_{z_{i}} Q_{i} \left( z_{i} \right) \cdot \log \frac{P \left( x_{i}, \ z_{i}; \ \theta \right)}{Q_{i} \left( z_{i} \right)}
  $$

- 等号成立当且仅当 $\forall \ z_{i} \rightarrow \frac{P \left( x_{i}, \ z_{i}; \ \theta \right)}{Q_{i} \left( z_{i} \right)} \equiv C$

- 由概率之和 $\sum_{z_{i}} Q_{i} \left( z_{i} \right) = 1$：

  $$
  \sum_{z_{i}} \frac{1}{C} P \left( x_{i}, \ z_{i}; \ \theta \right) = \frac{1}{C} P \left( x_{i}; \ \theta \right) = 1
  $$

- 由前一步推导结果：

  $$
  Q_{i} \left( z_{i} \right) = \frac{P \left( x_{i}, \ z_{i}; \ \theta \right)}{C} = \frac{P \left( x_{i}, \ z_{i}; \ \theta \right)}{P \left( x_{i}; \ \theta \right)} = P \left( z_{i}|x_{i}; \ \theta \right)
  $$

## 一般形式

### $E$ 步

$$
Q_{i} \left( z_{i} \right) = P \left( z_{i}|x_{i}; \ \theta \right)
$$

### $M$ 步

$$
\theta = \arg \max_{\theta} \sum_{i = 1}^{m} \sum_{z_{i}} \left( Q_{i} \left( z_{i} \right) \cdot \log \frac{P \left(x_{i}, \ z_{i}; \ \theta \right)}{Q_{i} \left( z_{i} \right)} \right)
$$