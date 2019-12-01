# $\mathrm{WGAN \ with \ Gradient \ Penalty}$

## 思想

- 通过梯度惩罚的方式，来强迫神经网络的权重满足 $\mathrm{K-Lipschitz}$ 连续

- $p_{r} \left( x \right)$ 表示样本的真实分布，$p_{z} \left( z \right)$ 表示噪声的先验分布，$p_{g} \left( x \right)$ 表示生成器学到的样本分布，$p_{s} \left( x \right)$ 表示真实分布与生成分布间的采样分布

- $f_{\omega} \left( x \right)$ 表示 $\mathrm{Lipschitz}$ 函数，$g_{\theta} \left( z \right)$ 表示对样本 $z$ 进行映射

- 生成器 $G$ 的优化目标是最小化真实分布与生成分布之间的 $\mathrm{Wasserstein}$ 距离，判别器 $D$ 的优化目标是最大化真实分布与生成分布之间的 $\mathrm{Wasserstein}$ 距离

- 优化目标：$\min \limits_{G} \max \limits_{D} V \left( D,\ G \right) = \mathbb{E}_{x \sim p_{r} \left( x \right)} \left[ f_{\omega} \left( x \right) \right] - \mathbb{E}_{x \sim p_{g} \left( x \right)} \left[ f_{\omega} \left( x \right) \right] - \lambda \mathbb{E}_{x \sim p_{s} \left( x \right)} \left[ \left( ||\nabla_{x} D \left( x \right)||_{2} - 1 \right)^{2} \right]$

## 原始 $\mathrm{WGAN}$ 缺点

### 不能充分利用网络容量

- 经过充分训练的神经网络，其权值基本分布在 $-c$ 和 $c$ 两个端点上

- 网络不能充分利用神经网络的容量，只能学到样本分布的简单近似

### 梯度消失或梯度爆炸

- 在不使用 $\mathrm{Batch \ Normalization}$ 时，截断常数 $c$ 不容易控制：

  - $c$ 值太大，导致后续层梯度爆炸

  - $c$ 值太小，导致后续层梯度消失

## 模型改善

- 通过对中间样本的梯度惩罚，来代替权值截断

- 由于单独惩罚每个样本的梯度，判别器不能使用 $\mathrm{Batch \ Normalization}$

- 可以使用基于动量的优化方法，比如 $\mathrm{Adam}$

## 数学推导

### $\mathrm{1-Lipschitz}$ 连续

- 当一个函数满足 $\mathrm{1-Lipschitz}$ 连续时，其梯度的范数不超过 $1$，即：

  $$
  D \in \mathrm{1-Lipschitz} \quad \leftrightarrow \quad ||\nabla_{x} D \left( x \right)|| \leq 1
  $$

- 在判别器最优时，对于真实样本 $x$ 和生成样本 $y$，其采样样本 $x_{t} = t \cdot x + \left( 1 - t \right) \cdot y$ 梯度的范数恒为 $1$，即：

  $$
  \nabla_{x} D^{*} \left( x_{t} \right) = \frac{y - x}{||y - x||}
  $$

- 为了使权重矩阵满足 $\mathrm{1-Lipschitz}$ 连续，可以约束采样样本梯度的期望值：

  $$
  \mathbb{E}_{x \sim p_{s} \left( x \right)} \left[ \left( ||\nabla_{x} D \left( x \right)||_{2} - 1 \right)^{2} \right]
  $$

## 训练过程

- 论文中，$\alpha = 0.0001, \ \beta_{1} = 0.5, \ \beta_{2} = 0.9, \ \lambda = 10, \ m = 64, \ k = 5$

- 每 $k$ 步更新 $D$，每 $1$ 步都更新 $G$

- 对于每一次迭代：

  - 对 $D$ 进行 $k$ 次迭代：

    - 从 $p_{r} \left( x \right)$ 中随机选择 $m$ 个样本

    - 从 $p_{z} \left( z \right)$ 中随机选择 $m$ 个样本

    - 生成 $m$ 对样本的中间样本：

      $$
      s_{i} = t \cdot x_{i} + \left( 1 - t \right) \cdot g_{\theta} \left( z_{i} \right)
      $$

    - 计算 $m$ 个损失值：

      $$
      L_{i} = f_{\omega} \left( x_{i} \right) - f_{\omega} \left( z_{i} \right) - \lambda \left[ \left( ||\nabla_{s} f_{\omega} \left( s_{i} \right)||_{2} - 1 \right)^{2} \right]
      $$

    - 通过梯度上升更新 $D$ 的权重：

      $$
      \delta \leftarrow \nabla_{\omega} \frac{1}{m} \sum_{i = 1}^{m} L_{i}
      $$

      $$
      \omega \leftarrow \omega + \mathrm{Adam} \left( \delta, \ \omega, \ \alpha, \ \beta_{1}, \ \beta_{2} \right)
      $$

  - 对 $G$ 进行迭代：

    - 从 $p_{z} \left( z \right)$ 中随机选择 $m$ 个样本

    - 通过梯度下降更新 $G$ 的权重：

      $$
      \delta \leftarrow -\nabla_{\theta} \frac{1}{m} \sum_{i = 1}^{m} f_{\omega} \left( g_{\theta} \left( z^{i} \right) \right)
      $$

      $$
      \theta \leftarrow \theta - \mathrm{Adam} \left( \delta, \ \theta, \ \alpha, \ \beta_{1}, \ \beta_{2} \right)
      $$