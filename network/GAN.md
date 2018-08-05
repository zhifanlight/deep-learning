<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Generative Adversarial Network

## 思想

- \\(p\_{r}(x)\\) 表示样本的真实分布，\\(p_{z}(z)\\) 表示噪声的先验分布，\\(p\_{g}(x)\\) 表示生成器学到的样本分布

- \\(D(x)\\) 表示样本 \\(x\\) 来自真实数据的概率，\\(G(z)\\) 表示对样本 \\(z\\) 进行映射

- 生成器 \\(G\\) 的优化目标是生成与真实数据分布尽可能相同的样本，判别器 \\(D\\) 的优化目标是将来自生成器和真实数据的样本分别开；当判别器 \\(D\\) 无法进行判别时达到全局最优，此时 \\(p\_{g}(x) = p\_{r}(x)\\)，判别器近似于随机猜测

- 优化目标：\\(\min\limits\_{G} \max\limits\_{D} V(D,G) = \mathbb{E}\_{x \sim p\_{r}(x)}[\log D(x)] + \mathbb{E}\_{z \sim p\_{z}(z)}[\log(1 - D(G(z)))]\\)

## 数学推导

- 在 \\(G\\) 固定时，\\(D\\) 的最优解是 \\(D^{*} = \frac{p\_{r}}{p\_{r} + p\_{g}}\\)；证明：

	$$
	\begin{aligned}
	target \qquad &\max\limits\_{D} V(D,G) \newline \newline
	V(D,G) &= \int\_{x} p\_{r}(x) \log(D(x)) dx + \int\_{z} p\_{z}(z) \log(1 - D(G(z))) dz \newline
	&= \int\_{x} p\_{r}(x) \log(D(x)) + p\_{g}(x) \log(1 - D(x)) dx \newline \newline
	prior \qquad &a \ln x + b \ln (1-x) \quad maximize \quad when \quad x=\frac{a}{a+b} \newline \newline
	\Rightarrow \qquad &V(D,G) \quad maximize \quad when \quad D = \frac{p\_{r}}{p\_{r} + p\_{g}} \newline \newline
	\end{aligned}
	$$

- 在 \\(D\\) 最优时，\\(G\\) 的最优解是 \\(p\_{g} = p_{data}\\)；证明：

	$$
	\begin{aligned}
	target \qquad &\min\limits\_{G} V(D,G) \newline \newline
	\qquad V(D^{*},G) &= \int\_{x} p\_{r}(x) \log(\frac{p\_{r}(x)}{p\_{r}(x)+p\_{g}(x)}) + p\_{g}(x) \log(\frac{p\_{g}(x)}{p\_{r}(x)+p\_{g}(x)}) dx \newline
	&= \int\_{x} p\_{r}(x) \log(\frac{1}{2}\frac{p\_{r}(x)}{\frac{p\_{r}(x)+p\_{g}(x)}{2}}) + p\_{g}(x) \log(\frac{1}{2}\frac{p\_{g}(x)}{\frac{p\_{r}(x)+p\_{g}(x)}{2}}) dx \newline
	&= -2\log2 + \int\_{x} p\_{r}(x) \log(\frac{p\_{r}(x)}{\frac{p\_{r}(x)+p\_{g}(x)}{2}}) + p\_{g}(x) \log(\frac{p\_{g}(x)}{\frac{p\_{r}(x)+p\_{g}(x)}{2}}) dx\newline
	&= -2\log2 + KL(p\_{r}||\frac{p\_{r}+p\_{g}}{2}) + KL(p\_{g}||\frac{p\_{r}+p\_{g}}{2}) \newline
	&= -2\log2 + 2JS(p\_{r}||p\_{g}) \newline \newline
	prior \qquad &JS(p\_{1}||p\_{2}) \quad minimize \quad when \quad p\_{1} = p\_{2} \newline \newline
	\Rightarrow \qquad &V(D^{\*},G) \quad minimize \quad when \quad p\_{g} = p\_{r} 
	\end{aligned}
	$$

## 训练过程

- 每 \\(k\\) 步更新 \\(D\\)，每 \\(1\\) 步都更新 \\(G\\)

- 对于每一次迭代：

	- 对 \\(D\\) 进行 \\(k\\) 次迭代：

		- 从 \\(p\_{r}(x)\\) 中随机选择 \\(m\\) 个样本

		- 从 \\(p\_{z}(z)\\) 中随机选择 \\(m\\) 个样本

		- 通过随机梯度上升更新 \\(D\\) 的权重：

			$$\nabla\_{\theta\_{d}} \frac{1}{m} \sum\_{i=1}^{m} [\log D(x^{i}) + \log(1 - D(G(z^{i})))]$$

	- 对 \\(G\\) 进行迭代：

		- 从 \\(p\_{z}(z)\\) 中随机选择 \\(m\\) 个样本

		- 通过随机梯度下降更新 \\(G\\) 的权重：

			$$\nabla\_{\theta\_{g}} \frac{1}{m} \sum\_{i=1}^{m} \log(1 - D(G(z^{i})))$$

## 优缺点

### 优点

- 可以训练任何一种生成器网络

- 通过无监督方式学习样本的数据分布来产生类似样本，过程类似于人类的学习

### 缺点

- 网络通常难以收敛

- 有可能导致生成器退化，进而导致训练无法继续