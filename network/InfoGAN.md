<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Information Maximizing GAN

## 思想

- 标准的 GAN 对输入噪声没有任何约束；如果输入噪声的某一维度对应一个语义特征，那么 GAN 将更具可解释性；通过改变这一维度上的值，可以定向生成更多样本

- 互信息描述变量间的独立程度，InfoGAN 通过最大化互信息，约束噪声的使用方式：

	- 把输入分为两类：噪声向量 \\(z\\)、语义向量 \\(c\\)

		- 噪声向量是标准 GAN 的输入噪声

		- 语义向量是相互独立的一系列隐含变量

	- 语义向量的信息，在生成样本的过程中不应该消失：

		- 即最大化 \\(c\\) 与样本 \\(G(z,c)\\) 的互信息 \\(I(c;G(z,c))\\)

- 优化目标：\\( \min\limits\_{G} \max\limits\_{D} V\_{I}(D,G) = V(G,D) - \lambda \ I(c;G(z,c))\\)

## 模型改善

- 把输入分为两类：噪声向量 \\(z\\)、语义向量 \\(c\\)

	- 对于离散型 \\(c\\)，判别器额外输出类别概率，以拟合多项式分布参数 \\(p\_{1}, p\_{2}, ..., p\_{n}\\)

	- 对于连续型 \\(c\\)，判别器额外输出方差和均值，以拟合高斯分布参数 \\(\sigma, \mu\\)

- 除 \\(\min\limits\_{G} \max\limits\_{D} V(D,G)\\) 外，最大化 \\(\mathbb{E} \left[log \ Q(c|x)\right]\\)，其中 \\(Q(c|x)\\) 是 \\(P(c|x)\\) 的近似

## 数学推导

### 互信息、\\(KL\\) 散度

- 参见 [Entropy.md](../basic/Entropy.md)

### 后验分布近似

- 直接计算 \\(I(c;G(z,c))\\) 时，需要用到后验分布 \\(P(c|x)\\)，而 \\(P(c|x)\\) 难以采样和估算；因此可以引入辅助分布 \\(Q(c|x)\\) 来近似 \\(P(c|x)\\)，此时 \\(I(c;G(z,c))\\) 计算如下：

	$$
	\begin{align\*}
	I(c;G(z,c)) &= H(c) - H(c|G(z,c)) \newline
	&= \sum\_{x \sim G(z, c)} \sum\_{c' \sim P(c)} P(c',x) \cdot log \ P(c'|x) + H(c) \newline
	&= \sum\_{x \sim G(z,c)} \left[ \sum\_{c' \sim P(c)} P(c',x) \cdot log \ \frac{P(c'|x)} {Q(c'|x)} + \sum\_{c' \sim P(c)} P(c',x) \cdot log \ Q(c'|x) \right] + H(c) \newline
	&= \sum\_{x \sim G(z,c)} \left[ \sum\_{c' \sim P(c)} P(x) \cdot P(c'|x) \cdot log \ \frac{P(c'|x)} {Q(c'|x) } + \sum\_{c' \sim P(c)} P(c',x) \cdot log \ Q(c'|x) \right] + H(c) \newline
	&= \sum\_{x \sim G(z,c)} \left[ P(x) \cdot KL\left(P(c'|x)||Q(c'|x)\right) + \sum\_{c' \sim P(c)} P(c',x) \cdot log \ Q(c'|x) \right] + H(c) \newline
	&\geq \sum\_{x \sim G(z,c)} \sum\_{c' \sim P(c)} P(c',x) \cdot log \ Q(c'|x) + H(c) \newline
	&= \mathbb{E}\_{x \sim G(z,c), \ c' \sim P(c)} \left[log \ Q(c'|x)\right] + H(c) \newline
	\end{align\*}
	$$

- 当 \\(Q(c'|x) = P(c'|x)\\) 时，上式等号成立；而对于给定分布，\\(H(c)\\) 是定值：

	$$ max \ I(c;G(z,c)) = max \ \mathbb{E}\_{x \sim G(z,c), \ c' \sim P(c)} \left[log \ Q(c'|x)\right] $$

## 训练过程

- 论文中，\\(\lambda = 1\\)

- 对于离散型分量，\\(Q(c|x)\\)用多项式分布表示：

	- 先用 Softmax 计算各项概率 \\(p\_{i}\\)，再计算多项式分布概率：

		$$ Q(c'|x) = \prod\_{i} p\_{i}^{\ \Phi(c'=i)} $$
		
	- 如果类别标签 \\(c'\\) 是 one-hot 向量，\\(log \ Q(c'|x)\\) 计算如下：

		$$ log \ Q(c'|x) = \sum\_{i} c'\_{i} \cdot log \ p\_{i} $$

- 对于连续型分量，\\(Q(c|x)\\)用高斯分布表示：

	- 直接计算高斯分布概率：

		$$ Q(c'|x) = \frac{1}{\sqrt{2\pi} \ \sigma} e^{-\frac{(c'-\mu)^{2}}{2\sigma^{2}}}$$
		
	- 当标准差 \\(\sigma\\) 固定时，其他项均为常量，\\(log \ Q(c'|x)\\) 计算如下：

		$$ log \ Q(c'|x) = -(c' - \mu) ^ {2} $$