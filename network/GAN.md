# $\mathrm{Generative \ Adversarial \ Network}$

## 思想

- $p_{r} \left( x \right)$ 表示样本的真实分布，$p_{z} \left( z \right)$ 表示噪声的先验分布，$p_{g} \left( x \right)$ 表示生成器学到的样本分布

- $D \left( x \right)$ 表示样本 $x$ 来自真实数据的概率，$G \left( z \right)$ 表示对样本 $z$ 进行映射

- 生成器 $G$ 的优化目标是生成与真实数据分布尽可能相同的样本，判别器 $D$ 的优化目标是将来自生成器和真实数据的样本分别开；当判别器 $D$ 无法进行判别时达到全局最优，此时 $p_{g} \left( x \right) = p_{r} \left( x \right)$，判别器近似于随机猜测

- 优化目标：$\min \limits_{G} \max \limits_{D} V \left( D, \ G \right) = \mathbb{E}_{x \sim p_{r} \left( x \right)} \left[ \log D \left( x \right) \right] + \mathbb{E}_{z \sim p_{z} \left( z \right)} \left[ \log \left( 1 - D \left( G \left( z \right) \right) \right) \right]$

## 数学推导

- 在 $G$ 固定时，$D$ 的最优解是 $D^{*} = \frac{p_{r}}{p_{r} + p_{g}}$；证明：

  $$
  \begin{aligned}
  \mathrm{target} \qquad &\max \limits_{D} V \left( D, \ G \right) \newline \newline
   V \left( D, \ G \right) &= \int_{x} p_{r} \left( x \right) \log \left( D \left( x \right) \right) \mathrm{d} x + \int_{z} p_{z} \left( z \right) \log \left( 1 - D \left( G \left( z \right) \right) \right) \mathrm{d}z \newline
  &= \int_{x} p_{r} \left( x \right) \log \left( D \left( x \right) \right) + p_{g} \left( x \right) \log \left( 1 - D \left( x \right) \right) \mathrm{d}x \newline \newline
  \mathrm{prior} \qquad &a \ln x + b \ln \left( 1 - x \right) \quad \mathrm{maximize \quad when} \quad x = \frac{a}{a + b} \newline \newline
  \Rightarrow \qquad &V \left( D, \ G \right) \quad \mathrm{maximize \quad when} \quad D = \frac{p_{r}}{p_{r} + p_{g}} \newline \newline
  \end{aligned}
  $$

- 在 $D$ 最优时，$G$ 的最优解是 $p_{g} = p_{\mathrm{data}}$；证明：

  $$
  \begin{aligned}
  \mathrm{target} \qquad &\min \limits_{G} V \left( D, \ G \right) \newline \newline
  \qquad V \left( D^{*}, \ G \right) &= \int_{x} p_{r} \left( x \right) \log \left( \frac{p_{r} \left( x \right)}{p_{r} \left( x \right) + p_{g} \left( x \right)} \right) + p_{g} \left( x \right) \log \left( \frac{p_{g} \left( x \right)}{p_{r} \left( x \right) + p_{g} \left( x \right)} \right) \mathrm{d}x \newline
  &= \int_{x} p_{r} \left( x \right) \log \left( \frac{1}{2} \frac{p_{r} \left( x \right)}{\frac{p_{r} \left( x \right) + p_{g} \left( x \right)}{2}} \right) + p_{g} \left( x \right) \log \left( \frac{1}{2} \frac{p_{g} \left( x \right)}{\frac{p_{r} \left( x \right) + p_{g} \left( x \right)}{2}} \right) \mathrm{d}x \newline
  &= -2 \log 2 + \int_{x} p_{r} \left( x \right) \log \left( \frac{p_{r} \left( x \right)}{\frac{p_{r} \left( x \right) + p_{g} \left( x \right)}{2}} \right) + p_{g} \left( x \right) \log \left( \frac{p_{g} \left( x \right)}{\frac{p_{r} \left( x \right) + p_{g} \left( x \right)}{2}} \right) \mathrm{d}x \newline
  &= -2 \log 2 + \mathrm{KL} \left( p_{r}||\frac{p_{r} + p_{g}}{2} \right) + \mathrm{KL} \left( p_{g}||\frac{p_{r} + p_{g}}{2} \right) \newline
  &= -2 \log 2 + 2 \mathrm{JS} \left( p_{r}||p_{g} \right) \newline \newline
  \mathrm{prior} \qquad &\mathrm{JS} \left( p_{1}||p_{2} \right) \quad \mathrm{minimize \quad when } \quad p_{1} = p_{2} \newline \newline
  \Rightarrow \qquad &V \left( D^{*}, \ G \right) \quad \mathrm{minimize \quad when} \quad p_{g} = p_{r}
  \end{aligned}
  $$

## 训练过程

- 每 $k$ 步更新 $D$，每 $1$ 步都更新 $G$

- 对于每一次迭代：

  - 对 $D$ 进行 $k$ 次迭代：

    - 从 $p_{r} \left( x \right)$ 中随机选择 $m$ 个样本

    - 从 $p_{z} \left( z \right)$ 中随机选择 $m$ 个样本

    - 通过随机梯度上升更新 $D$ 的权重：

      $$
      \nabla_{\theta_{d}} \frac{1}{m} \sum_{i = 1}^{m} \left[ \log D \left( x^{i} \right) + \log \left( 1 - D \left( G \left( z^{i} \right) \right) \right) \right]
      $$

  - 对 $G$ 进行迭代：

    - 从 $p_{z} \left( z \right)$ 中随机选择 $m$ 个样本

    - 通过随机梯度下降更新 $G$ 的权重：

      $$
      \nabla_{\theta_{g}} \frac{1}{m} \sum_{i = 1}^{m} \log \left( 1 - D \left( G \left( z^{i} \right) \right) \right)
      $$

## 优缺点

### 优点

- 可以训练任何一种生成器网络

- 通过无监督方式学习样本的数据分布来产生类似样本，过程类似于人类的学习

### 缺点

- 网络通常难以收敛

- 有可能导致生成器退化，进而导致训练无法继续