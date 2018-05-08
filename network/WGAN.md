<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Wasserstein GAN

## 思想

- \\(p\_{r}(x)\\) 表示样本的真实分布，\\(p_{z}(z)\\) 表示噪声的先验分布，\\(p\_{g}(x)\\) 表示生成器学到的样本分布

- \\(f\_{\omega}(x)\\) 表示 Lipschitz 函数，\\(g\_{\theta}(z)\\) 表示对样本 \\(z\\) 进行映射

- 生成器 \\(G\\) 的优化目标是最小化真实分布与生成分布之间的 Wasserstein 距离，判别器 \\(D\\) 的优化目标是最大化真实分布与生成分布之间的 Wasserstein 距离

- 优化目标：\\(\min\limits\_{G} \max\limits\_{D} V(D,G) = \mathbb{E}\_{x \sim p\_{r}(x)}[f\_{\omega}(x)] - \mathbb{E}\_{x \sim p\_{g}(x)}[f\_{\omega}(x)]\\)

## 原始 GAN 缺点

### 梯度消失

- 容易落入判别器的 sigmoid 饱和区间，导致梯度消失

### 模式崩溃

- 不使用 logD trick，训练前期容易导致梯度消失

- 使用 logD trick，容易导致模式崩溃

- 最优判别器如下：

	$$ D^{\*}(x) = \frac{p\_{r}(x)}{p\_{r}(x) + p\_{g}(x)} $$

- 此时 \\(G\_{loss}\\) 计算如下：

	$$ \mathbb{E}\_{x \sim p\_{r}(x)}[\log D^{\*}(x)] + \mathbb{E}\_{x \sim p\_{g}(x)}[\log(1 - D^{\*}(x))] = 2JS(p\_{r}||p\_{g}) - 2\log2 $$

- 对 \\(KL(p\_{g}||p\_{r})\\) 做如下变换：

	$$
	\begin{aligned}
	KL(p\_{g}||p\_{r}) &= \mathbb{E}\_{x \sim p\_{g}(x)} \left[\log\frac{p\_{g}(x)}{p\_{r}(x)}\right] \newline
	&= \mathbb{E}\_{x \sim p\_{g}(x)} \left[\log\frac{\frac{p\_{g}(x)}{p\_{g}(x) + p\_{r}(x)}}{\frac{p\_{r}(x)}{p\_{g}(x) + p\_{r}(x)}}\right] \newline
	&= \mathbb{E}\_{x \sim p\_{g}(x)} \left[\log\frac{1 - D^{\*}(x)}{D^{\*}(x)}\right] \newline
	&= \mathbb{E}\_{x \sim p\_{g}(x)} \left[\log(1 - D^{\*}(x))\right] - \mathbb{E}\_{x \sim p\_{g}(x)} \left[\log D^{\*}(x)\right] \newline
	\end{aligned}
	$$

- 综合上式可得：

	$$
	\begin{aligned}
	\mathbb{- \ E}\_{x \sim p\_{g}(x)} \left[\log D^{\*}(x)\right] &= KL(p\_{g}||p\_{r}) - \mathbb{E}\_{x \sim p\_{g}(x)} \left[\log(1 - D^{\*}(x))\right] \newline
	&= KL(p\_{g}||p\_{r}) - [2JS(p\_{r}||p\_{g}) - 2\log2 - \mathbb{E}\_{x \sim p\_{r}}[\log D^{\*}(x)]] \newline
	&= KL(p\_{g}||p\_{r}) - 2JS(p\_{r}||p\_{g}) + Const \newline
	\end{aligned}
	$$

- 最小化 \\(- 2JS(p\_{r}||p\_{g})\\) 意味着最大化 \\(p\_{r}\\) 与 \\(p\_{g}\\) 之间的差距，导致生成器不稳定

- \\(KL\\) 散度的不对称性，会导致模式崩溃：

	$$ \\left\\{ \begin{matrix} p\_{g}(x)\log\frac{p\_{g}(x)}{p\_{r}(x)} \rightarrow 0 \quad & if \ p\_{g} \rightarrow 0, \ p\_{r} \rightarrow 1 \\\\ p\_{g}(x)\log\frac{p\_{g}(x)}{p\_{r}(x)} \rightarrow +\infty & if \ p\_{g} \rightarrow 1, \ p\_{r} \rightarrow 0 \end{matrix} \\right\. $$
			
	- 前者表示生成器没能生成数据集中存在的样本

	- 后者表示生成器生成了数据集中不存在的样本

	- 由于惩罚代价不同，生成器尽量生成一些重复但“安全”的样本，也不生成不存在的样本，最终导致了模式崩溃

## 模型改善

- 由于是回归任务，判别器最后一层去掉 Sigmoid

- 生成器和判别器的损失函数不取 log

- 每次更新判别器的参数之后，将其截断到固定范围内

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