<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# WGAN with Gradient Penalty

## 思想

- 通过梯度惩罚的方式，来强迫神经网络的权重满足 K-Lipschitz 连续

- \\(p\_{r}(x)\\) 表示样本的真实分布，\\(p\_{z}(z)\\) 表示噪声的先验分布，\\(p\_{g}(x)\\) 表示生成器学到的样本分布，\\(p\_{s}(x)\\) 表示真实分布与生成分布间的采样分布

- \\(f\_{\omega}(x)\\) 表示 Lipschitz 函数，\\(g\_{\theta}(z)\\) 表示对样本 \\(z\\) 进行映射

- 生成器 \\(G\\) 的优化目标是最小化真实分布与生成分布之间的 Wasserstein 距离，判别器 \\(D\\) 的优化目标是最大化真实分布与生成分布之间的 Wasserstein 距离

- 优化目标：\\( \min\limits\_{G} \max\limits\_{D} V(D,G) = \mathbb{E}\_{x \sim p\_{r}(x)}[f\_{\omega}(x)] - \mathbb{E}\_{x \sim p\_{g}(x)}[f\_{\omega}(x)] - \lambda \ \mathbb{E}\_{x \sim p\_{s}(x)} \left[ \left( ||\nabla\_{x}D(x)||\_{2} - 1 \right) ^ {2} \right] \\)

## 原始 WGAN 缺点

### 不能充分利用网络容量

- 经过充分训练的神经网络，其权值基本分布在 -c 和 c 两个端点上

- 网络不能充分利用神经网络的容量，只能学到样本分布的简单近似

### 梯度消失或梯度爆炸

- 在不使用 Batch Normalization 时，截断常数 c 不容易控制：

	- c 值太大，导致后续层梯度爆炸

	- c 值太小，导致后续层梯度消失

## 模型改善

- 通过对中间样本的梯度惩罚，来代替权值截断

- 由于单独惩罚每个样本的梯度，判别器不能使用 Batch Normalization

- 可以使用基于动量的优化方法，比如 Adam

## 数学推导

### 1-Lipschitz 连续

- 当一个函数满足 1-Lipschitz 连续时，其梯度的范数不超过 1，即：

	$$ D \in 1-Lipschitz \quad \leftrightarrow \quad ||\nabla\_{x}D(x)|| \leq 1 $$

- 在判别器最优时，对于真实样本 \\(x\\) 和生成样本 \\(y\\)，其采样样本 \\(x\_{t} = t \cdot x + (1 - t) \cdot y\\) 梯度的范数恒为 1，即：

	$$ \nabla\_{x}D^{*}(x\_{t}) = \frac{y - x}{||y - x||} $$

- 为了使权重矩阵满足 1-Lipschitz 连续，可以约束采样样本梯度的期望值：

	$$ \mathbb{E}\_{x \sim p\_{s}(x)} \left[ \left( ||\nabla\_{x}D(x)||\_{2} - 1 \right) ^ {2} \right] $$

## 训练过程

- 论文中，\\(\alpha = 0.0001, \ \beta\_{1} = 0.5, \ \beta\_{2} = 0.9, \ \lambda = 10, \ m = 64, \ k = 5\\)

- 每 \\(k\\) 步更新 \\(D\\)，每 \\(1\\) 步都更新 \\(G\\)

- 对于每一次迭代：

	- 对 \\(D\\) 进行 \\(k\\) 次迭代：

		- 从 \\(p\_{r}(x)\\) 中随机选择 \\(m\\) 个样本

		- 从 \\(p\_{z}(z)\\) 中随机选择 \\(m\\) 个样本

		- 生成 \\(m\\) 对样本的中间样本：

			$$s\_{i} = t \cdot x\_{i} + (1 - t) \cdot g\_{\theta}(z\_{i})$$
			
		- 计算 \\(m\\) 个损失值：

			$$L\_{i} = f\_{\omega}(x\_{i}) - f\_{\omega}(z\_{i}) - \lambda \left[ \left( ||\nabla\_{s}f\_{\omega}(s\_{i})||\_{2} - 1 \right) ^ {2} \right]$$

		- 通过梯度上升更新 \\(D\\) 的权重：

			$$\delta \leftarrow \nabla\_{\omega} \frac{1}{m} \sum\_{i=1}^{m} L\_{i}$$

			$$\omega \leftarrow \omega + Adam(\delta, \omega, \alpha, \beta\_{1}, \beta\_{2})$$

	- 对 \\(G\\) 进行迭代：

		- 从 \\(p\_{z}(z)\\) 中随机选择 \\(m\\) 个样本

		- 通过梯度下降更新 \\(G\\) 的权重：

			$$\delta \leftarrow -\nabla\_{\theta} \frac{1}{m} \sum\_{i=1}^{m} f\_{\omega}(g\_{\theta}(z^{i}))$$
			
			$$\theta \leftarrow \theta - Adam(\delta, \theta, \alpha, \beta\_{1}, \beta\_{2})$$