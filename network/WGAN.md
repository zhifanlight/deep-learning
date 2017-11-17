<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Wasserstein GAN

## 思想

- \\(p\_{r}(x)\\) 表示样本的真实分布，\\(p_{z}(z)\\) 表示噪声的先验分布，\\(p\_{g}(x)\\) 表示生成器学到的样本分布

- \\(f\_{\omega}(x)\\) 表示 Lipschitz 函数，\\(g\_{\theta}(z)\\) 表示对样本 \\(z\\) 进行映射

- 生成器 \\(G\\) 的优化目标是最小化真实分布与生成分布之间的 Wasserstein 距离，判别器 \\(D\\) 的优化目标是最大化真实分布与生成分布之间的 Wasserstein 距离

- 优化目标：\\(\min\limits\_{G} \max\limits\_{D} V(D,G) = \mathbb{E}\_{x \sim p\_{r}(x)}[f\_{\omega}(x)] - \mathbb{E}\_{x \sim p\_{g}(x)}[f\_{\omega}(x)]\\)

## 原始 GAN 缺点

#### 形式一：\\(G\_{loss} = \mathbb{E}\_{z \sim p\_{z}(z)}[log(1 - D(G(z)))]\\)

- 判别器训练的不好不坏才行，容易导致梯度消失，不容易控制：

	- 由于真实分布、生成分布基本不重叠，\\(JS(p\_{r}||p\_{g}) = log2\\)（参考 [DistanceMetrics.md] (../basic/DistanceMetrics.md)）

	- 当判别器最优时，\\(V(D^{*},G) = 2JS(p\_{r}||p\_{g}) - 2log2\\)（参考 [GAN.md] (GAN.md)）

	- 梯度近似为 \\(0\\)，导致生成器的训练无法继续

	- 判别器训练的不好，生成器梯度不准，到处乱跑

	- 判别器训练的太好，生成器梯度消失，无法继续

#### 形式二：\\(G\_{loss} = \mathbb{E}\_{z \sim p\_{z}(z)}[-log(D(G(z)))]\\)

- 由于度量距离的不合理，导致生成器不稳定，以及样本多样性不足：

	- 在最优判别器下（参考 [GAN.md] (GAN.md)）：

	$$\mathbb{E}\_{x \sim p\_{r}(x)}[logD^{\*}(x)] + \mathbb{E}\_{x \sim p\_{g}(x)}[log(1 - D^{\*}(x))] = 2JS(p\_{r}||p\_{g}) - 2log2$$
	
	$$D^{\*}(x) = \frac{p\_{r}(x)}{p\_{r}(x) + p\_{g}(x)}$$

	- 而 \\(KL(p\_{g}||p\_{r})\\) 可做如下变形：

	$$
	\begin{align\*}
	KL(p\_{g}||p\_{r}) &= \mathbb{E}\_{x \sim p\_{g}(x)} \left[log\frac{p\_{g}(x)}{p\_{r}(x)}\right] \newline
	&= \mathbb{E}\_{x \sim p\_{g}(x)} \left[log\frac{\frac{p\_{g}(x)}{p\_{g}(x) + p\_{r}(x)}}{\frac{p\_{r}(x)}{p\_{g}(x) + p\_{r}(x)}}\right] \newline
	&= \mathbb{E}\_{x \sim p\_{g}(x)} \left[log\frac{1 - D^{\*}(x)}{D^{\*}(x)}\right] \newline
	&= \mathbb{E}\_{x \sim p\_{g}(x)} \left[log(1 - D^{\*}(x))\right] - \mathbb{E}\_{x \sim p\_{g}(x)} \left[logD^{\*}(x)\right] \newline
\end{align\*}
	$$
	
	- 综合上式得：

	$$
	\begin{align\*}
	\mathbb{E}\_{x \sim p\_{g}(x)} \left[-logD^{\*}(x)\right] &= KL(p\_{g}||p\_{r}) - \mathbb{E}\_{x \sim p\_{g}(x)} \left[log(1 - D^{\*}(x))\right] \newline
	&= KL(p\_{g}||p\_{r}) - [2JS(p\_{r}||p\_{g}) - 2log2 - \mathbb{E}\_{x \sim p\_{g}(x)}[log(1 - D^{\*}(x))]] \newline
	&= KL(p\_{g}||p\_{r}) - 2JS(p\_{r}||p\_{g}) + 2log2 + \mathbb{E}\_{x \sim p\_{g}(x)}[log(1 - D^{\*}(x))] \newline
\end{align\*}
	$$
	
	- 后两项与生成器无关，最小化 \\(G\_{loss}\\) 意味着最小化 \\(KL(p\_{g}||p\_{r}) - 2JS(p\_{r}||p\_{g})\\)：

		- 减去 \\(JS\\) 散度，导致本该最小化的过程不稳定，进而导致生成器不稳定

		- 而 \\(KL\\) 散度的不对称性，导致了样本的多样性不足，即 collapse mode：

			$$
			\\left\\{ \begin{matrix} p\_{g}(x)log\frac{p\_{g}(x)}{p\_{r}(x)} \rightarrow 0 & if \ p\_{g} \rightarrow 0, \ p\_{r} \rightarrow 1 \\\\ p\_{g}(x)log\frac{p\_{g}(x)}{p\_{r}(x)} \rightarrow +\infty & if \ p\_{g} \rightarrow 1, \ p\_{r} \rightarrow 0 \end{matrix} \\right\.
			$$
			
			- 前者对应没能生成真实样本，后者对应生成了不真实的样本

			- 由于惩罚代价不同，生成器尽可能少生成真实样本，也不生成不真实样本，导致了样本多样性不足

## 模型改善

- 由于是回归任务，判别器最后一层去掉 Sigmoid

- 生成器和判别器的损失函数不取对数

- 每次更新判别器的参数之后，将其截断到固定范围内

- 不要用基于动量的优化算法，推荐 RMSProp

## 数学推导

- Lipschitz 连续：

	$$|f(x\_{1}) - f(x\_{2})| \leq K \cdot |x\_{1} - x\_{2}|$$
	
	- 即导数的绝对值不超过 \\(K\\)

- Wasserstein 距离中的 \\(\inf \limits\_{\gamma \sim \Pi(p,q)}\\) 无法直接求解（参考 [DistanceMetrics.md] (../basic/DistanceMetrics.md)）

- 将 Wasserstein 距离转换为：

	$$K \cdot W(P\_{r}, P\_{g}) = \sup \limits \_{||f||\_{L} \leq K} \mathbb{E}\_{x \sim p\_{r}(x)}[f(x)] - \mathbb{E}\_{x \sim p\_{g}(x)}[f(x)]$$
	
	- 其中 \\(sup\\) 表示最小上界

- 上式表示在函数 \\(f\\) 的 Lipchitz 常数不超过 \\(K\\) 的条件下，对所有满足条件的 \\(f\\)，取到 \\(\mathbb{E}\_{x \sim p\_{r}(x)}[f(x)] - \mathbb{E}\_{x \sim p\_{g}(x)}[f(x)]\\) 的上界，即：
	
	$$K \cdot W(P\_{r}, P\_{g}) \approx \max \limits \_{\omega: ||f\_{\omega}||\_{L} \leq K} \mathbb{E}\_{x \sim p\_{r}(x)}[f(x)] - \mathbb{E}\_{x \sim p\_{g}(x)}[f(x)]$$
	
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