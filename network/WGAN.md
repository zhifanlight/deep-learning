<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Wasserstein GAN

## 思想

- \\(p\_{r}(x)\\) 表示样本的真实分布，\\(p_{z}(z)\\) 表示噪声的先验分布，\\(p\_{g}(x)\\) 表示生成器学到的样本分布

- \\(f\_{\omega}(x)\\) 表示 Lipschitz 函数，\\(g\_{\theta}(z)\\) 表示对样本 \\(z\\) 进行映射

- 生成器 \\(G\\) 的优化目标是最小化真实分布与生成分布之间的 Wasserstein 距离，判别器 \\(D\\) 的优化目标是最大化真实分布与生成分布之间的 Wasserstein 距离

- 优化目标：\\(\min\limits\_{G} \max\limits\_{D} V(D,G) = \mathbb{E}\_{x \sim p\_{r}(x)}[f\_{\omega}(x)] - \mathbb{E}\_{x \sim p\_{g}(x)}[f\_{\omega}(x)]\\)

## 原始 GAN 缺点

### 训练不稳定

- 不使用 logD trick

	- 判别器训练的太好

		- 很容易区分真实分布和生成分布，\\(JS\\) 散度为定值，梯度基本为零

	- 判别器训练的不好

		- 提供给生成器的梯度不准，训练过程容易震荡

### 模式崩溃

- 使用 logD trick，生成器目标函数如下：

	$$ KL(P\_{g} || P\_{r}) - 2 \cdot JS(P\_{g} || P\_{r}) $$

	- 要同时最大化、最小化两个分布的距离，导致训练不稳定

	- 由于 \\(KL\\) 散度不对称性：

		$$ \lim\_{p\_{g} \rightarrow 0, \ p\_{r} > 0} p\_{g} \log \frac{p\_{g}}{p\_{r}} \rightarrow 0 \quad \lim\_{p\_{g} > 0, \ p\_{r} \rightarrow 0} p\_{g} \log \frac{p\_{g}}{p\_{r}} \rightarrow \infty $$
		
		- 前者表示没能生成真实分布中存在的样本，后者表示生成了真实分布中不存在的样本

	- 理想情况下：

		$$ \lim\_{p\_{g} = p\_{r}} p\_{g} \log \frac{p\_{g}}{p\_{r}} = 0 $$
		
		- 即生成分布与真实分布完全一致

		- 真实分布来自采样，生成分布来自随机噪声；训练过程中，生成分布的样本数远远大于真实分布的样本数，两种分布不可能完全一致

	- 如果生成太多真实分布中不存在的样本，很容易导致第二种惩罚

	- 由于生成器总要生成样本，而生成分布与真实分布无法完全一致，生成器会倾向于重复生成真实分布中存在的样本，导致模式崩溃

## 模型改善

- 由于是回归任务，判别器最后一层去掉 Sigmoid

- 生成器和判别器的损失函数不取 log

- 每次更新判别器的参数后，将其截断到固定范围内

- 不要用基于动量的优化算法，推荐 RMSProp

## 数学推导

- Lipschitz 连续：

	$$|f(x\_{1}) - f(x\_{2})| \leq K \cdot |x\_{1} - x\_{2}|$$
	
	- 即导数的绝对值不超过 \\(K\\)

- Wasserstein 距离中的 \\(\inf \limits\_{\gamma \sim \Pi(p,q)}\\) 无法直接求解（参考 [DistanceMetrics.md] (../basic/DistanceMetrics.md)）

- 将 Wasserstein 距离转换为：

	$$ W(P\_{r}, P\_{g}) = \sup \limits \_{||f||\_{L} \leq K} \frac{1}{K} \left[ \mathbb{E}\_{x \sim p\_{r}(x)}[f(x)] - \mathbb{E}\_{x \sim p\_{g}(x)}[f(x)] \right] $$
	
	- 其中 \\(sup\\) 表示最小上界

- 上式表示在函数 \\(f\\) 的 Lipchitz 常数不超过 \\(K\\) 的条件下，对所有满足条件的 \\(f\\)，取到 \\(\mathbb{E}\_{x \sim p\_{r}(x)}[f(x)] - \mathbb{E}\_{x \sim p\_{g}(x)}[f(x)]\\) 的上界，即：
	
	$$ W(P\_{r}, P\_{g}) \approx \max \limits \_{\omega: ||f\_{\omega}||\_{L} \leq K} \frac{1}{K} \left[ \mathbb{E}\_{x \sim p\_{r}(x)}[f(x)] - \mathbb{E}\_{x \sim p\_{g}(x)}[f(x)] \right] $$
	
	- 其中 \\(f\_{\omega}\\) 表示用参数为 \\(\omega\\) 的神经网络表示函数 \\(f\\)

- 由于 Lipchitz 连续的限制，以及 \\(K\\) 不影响梯度方向的事实，需要将神经网络 \\(f\_{\theta}\\) 的所有参数 \\(\omega\_{i}\\)限制在某个范围 \\([-c, c]\\)

## 训练过程

- 论文中，\\(\alpha = 0.00005, \ c = 0.01, \ m = 64, \ k = 5\\)

- 每 \\(k\\) 步更新 \\(D\\)，每 \\(1\\) 步都更新 \\(G\\)

- 对于每一次迭代：

	- 对 \\(D\\) 进行 \\(k\\) 次迭代：

		- 从 \\(p\_{r}(x)\\) 中随机选择 \\(m\\) 个样本

		- 从 \\(p\_{z}(z)\\) 中随机选择 \\(m\\) 个样本

		- 通过梯度上升更新 \\(D\\) 的权重：

			$$\delta \leftarrow \nabla\_{\omega} \frac{1}{m} \sum\_{i=1}^{m} [f\_{\omega}(x^{i}) - f\_{\omega}(g\_{\theta}(z^{i}))]$$

			$$\omega \leftarrow \omega + \alpha \cdot RMSProp(\omega, \delta)$$
			
			$$\omega \leftarrow clip(\omega, -c, c) $$

	- 对 \\(G\\) 进行迭代：

		- 从 \\(p\_{z}(z)\\) 中随机选择 \\(m\\) 个样本

		- 通过梯度下降更新 \\(G\\) 的权重：

			$$\delta \leftarrow -\nabla\_{\theta} \frac{1}{m} \sum\_{i=1}^{m} f\_{\omega}(g\_{\theta}(z^{i}))$$
			
			$$\theta \leftarrow \theta - \alpha \cdot RMSProp(\theta, \delta)$$