<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 概率论基础

## 期望

- 随机变量 \\(x\\) 的期望计算如下：

	- 离散值：

		$$ E[x] = \sum\_{x} P(x) \cdot x $$
		
	- 连续值：

		$$ E[x] = \int P(x) \cdot x \ dx $$

- 线性运算的期望等于期望的线性运算：

	$$ E \left( \sum\_{i=1}^{n} a\_{i} x\_{i} + c \right) = \sum\_{i=1}^{n} a\_{i} E(x\_{i}) + c $$

- 如果两个变量相互独立，乘积的期望等于期望的乘积：

	$$ E(xy) = E(x) \cdot E(y) $$

## 方差

- 随机变量 \\(x\\) 的方差计算如下：

	- 离散值：

		$$ Var(x) = E \left[ \ \left( x - E[x] \right)^{2} \ \right] $$

		- 进一步展开可得：

			$$ Var(x) = E [x^{2}] - E^{2}[x] $$

	- 连续值：

		$$ Var(x) = \int P(x) \cdot \left( x - E[x] \right)^{2} \ dx $$
		
		- 进一步展开可得：

			$$ Var(x) = E [x^{2}] - E^{2}[x] $$

- 线性运算的方差，等于所有协方差的线性运算：

	$$ Var \left( \sum\_{i=1}^{n} a\_{i} x\_{i} \right) = \sum\_{i=1}^{n} \sum\_{j=1}^{n} a\_{i} a\_{j} Cov(x\_{i}, x\_{j}) $$

- 如果两个变量相互独立，线性运算的方差计算如下：

	$$ Var(ax+by) = a^{2} Var(x) + b^{2} Var(y) $$

- 如果两个变量相互独立，且均值为 \\(0\\)，乘积的方差计算如下：

	$$ Var(xy) = Var(x) \cdot Var(y) $$

## 条件概率链式法则

- 多维随机变量的联合概率分布，可以分解成只有一个变量的条件概率相乘的形式：

	$$ P(a,b,c) = P(a|b,c) \cdot P(b|c) \cdot P(c) $$

	- 证明如下：

		$$ P(a,b,c) = P(a|b,c) \cdot P(b,c) = P(a|b,c) \cdot P(b|c) \cdot P(c) $$

## 常用概率分布

### 伯努利分布

- 随机变量 \\(x\\) 的取值集合为 \\(\\{0, \ 1\\}\\)，对应概率分别为 \\(1-\phi, \ \phi\\)

	$$ P(x) = \phi^{x} \cdot (1-\phi)^{1-x} $$

- 期望：

	$$ E[x] = 1 \cdot \phi + 0 \cdot (1 - \phi) = \phi $$
	
- 方差：

	$$ Var(x) = E[x^{2}] - E^{2}[x] = \phi(1 - \phi) $$
	
	- 其中 \\(E[x^{2}]\\) 计算如下：

		$$ E[x^{2}] = 1^{2} \cdot \phi + 0^{2} \cdot (1 - \phi) = \phi $$

### 均匀分布

- 随机变量 \\(x\\) 落在区间 \\([a, \ b]\\) 上任一点的概率均为 \\(\frac{1}{b-a}\\)

- 期望：

	$$ E[x] = \int\_{a}^{b} x \cdot \frac{1}{b-a} \ dx = \frac{a+b}{2} $$

- 方差：

	$$ Var(x) = E[x^{2}] - E^{2}[x] = \frac{(a-b)^{2}}{12} $$

	- 其中 \\(E[x^{2}]\\) 计算如下：

		$$ E[x^{2}] = \int\_{a}^{b} P(x) \cdot x^{2} \ dx = \frac{a^{2}+b^{2}+ab}{3} $$

### 高斯分布

- 又称正态分布，参考 [GaussianModels.md](GaussianModels.md)

### 泊松分布

- 描述单位时间（或空间）内随机事件发生次数的概率分布

- 假设单位时间内事件 \\(x\\) 发生 \\(\lambda\\) 次，则单位时间内，事件 \\(x\\) 发生 \\(k\\) 次的概率为：

	$$ P(x=k) = \frac{\lambda^{k} \cdot e^{-\lambda}}{k!} $$

- 在 \\(t\\) 个单位时间内，事件 \\(x\\) 发生 \\(k\\) 次的概率为：

	$$ P(x=k) = \frac{ \left( \lambda t \right)^{k} \cdot e^{-\lambda t}}{k!} $$

- 期望、方差均为 \\(\lambda\\)

	- 计算时用到 \\(e^{x}\\) 在 \\(x=0\\) 处的泰勒展开式

- 泊松分布是非对称分布，当 \\(\lambda\\) 较小时呈偏态分布；当 \\(\lambda=20\\) 时，近似为高斯分布

### 指数分布

- 描述两件事情发生的时间间隔的概率分布

- 由泊松分布推导而来，假设单位时间内事件 \\(x\\) 发生 \\(\lambda\\) 次

- 事件的时间间隔为 \\(t\\)，等价于在间隔 \\(t\\) 内事件 \\(x\\) 发生 \\(0\\) 次，代入泊松分布概率公式：

	$$ P(x>t) = P(x=0) = e^{-\lambda t} $$

	- 相应地，在间隔 \\(t\\) 内事件 \\(x\\) 发生的概率为：

		$$ P(x \leq t) = 1 - e^{-\lambda t} $$

	- 概率密度函数计算如下：

		$$ P(t; \lambda) = \lambda \cdot e^{-\lambda t} $$

- 期望为 \\(\frac{1}{\lambda}\\)，方差为 \\(\frac{1}{\lambda^{2}}\\)

	- 计算时用到分步积分法

## 大数定律

- 对于 \\(n\\) 个独立同分布的样本，当 \\(n \rightarrow \infty\\) 时，样本均值收敛到期望值：

	$$ \lim\_{n \rightarrow \infty}{\frac{1}{n} \sum\_{i=1}^{n} x\_{i}} = \mu $$

## 中心极限定理

- 对于 \\(n\\) 个独立同分布的样本，当 \\(n \rightarrow \infty\\) 时，样本均值近似服从正态分布：

	$$ \lim\_{n \rightarrow \infty} \frac{1}{n} \sum\_{i=1}^{n} x\_{i} \ \sim \ N(\mu, \frac{\sigma^{2}}{n}) $$

	- 其中 \\(\mu\\) 是原始样本均值，\\(\sigma^{2}\\) 是原始样本方差

	- 归一化得到：

		$$ \lim\_{n \rightarrow \infty} \frac{\frac{1}{n} \sum\_{i=1}^{n} x\_{i} - \mu}{\sigma \ / \sqrt{n}} \sim \ N(0, 1) $$