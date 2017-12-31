<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 广义线性模型

## 背景介绍

## 指数族分布

- 指数族分布具有以下形式：
	
	$$ p(y;\eta) = b(y) \cdot exp({\eta^{T}T(y) - a(\eta)}) $$
	
	- \\(\eta\\) 是分布的参数

	- \\(T(y)\\) 是 \\(y\\) 的函数，通常 \\(T(y)=y\\)

	- \\(a(\eta)\\) 是归一化因子，保证 \\(\sum p(y;\eta)=1\\)

### 伯努利分布

- 已知 \\(p(y;\phi) \sim B(\phi)\\)，其中 \\(\phi\\) 表示 \\(y=1\\) 的概率

- 对于伯努利分布，\\(b(y)=1, \ \eta=log\frac{\phi}{1-\phi}, \ T(y)=y, \ a(\eta)=log \ (1+e^{\eta})\\)

	$$
	\begin{align\*}
	p(y;\phi) &= \phi^{y} \cdot (1-\phi)^{1-y} \newline
	&= exp(log \ \phi^{y} \cdot (1-\phi)^{1-y}) \newline
	&= exp(y \cdot log \ \phi + (1-y) \cdot log \ (1-\phi)) \newline
	&= exp \left(y \cdot log \frac{\phi}{1-\phi} + log \ (1-\phi) \right)
	\end{align\*}
	$$
	
	- 由 \\(\eta=log\frac{\phi}{1-\phi}\\) 可导出 sigmoid 函数：
	
		$$ \phi=\frac{1}{1+e^{-\eta}} $$
	
	- 将 \\(\phi\\) 代入 \\(log \ (1-\phi)\\) 可得：
	
		$$ p(y;\phi) = exp \left(y \cdot log \frac{\phi}{1-\phi} - log \ (1+e^{\eta}) \right) $$

### 高斯分布

- 已知 \\(p(y;\phi) \sim N(\mu,\sigma^{2})\\)，其中 \\(\phi=(\mu,\sigma^{2})\\)

- 对于高斯分布，\\(b(y)=\frac{1}{\sqrt{2\pi} \sigma}exp \left( -\frac{y^{2}}{2\sigma^{2}} \right), \ \eta=\frac{\mu}{\sigma^{2}}, \ T(y)=y, \ a(\eta)=\frac{\eta^{2}\sigma^{2}}{2}\\)

	$$
	\begin{align\*}
	p(y;\phi) &= \frac{1}{\sqrt{2\pi}\sigma}exp \left( -\frac{(y-\mu)^2}{2\sigma^{2}} \right) \newline
	&= \frac{1}{\sqrt{2\pi}\sigma}exp \left( \frac{2y\mu-y^{2}-\mu^{2}}{2\sigma^{2}} \right) \newline
	&= \frac{1}{\sqrt{2\pi}\sigma}exp \left( -\frac{y^{2}}{2\sigma^{2}} \right) exp \left( \frac{\mu}{\sigma^{2}}y - \frac{\mu^{2}}{2\sigma^{2}}\right) \newline
	\end{align\*}
	$$

### 多项式分布

- 已知 \\(p(y;\phi) \sim M(\phi\_{1},\phi\_{2},...,\phi\_{K})\\)，其中 \\(\phi\_{i}\\) 表示 \\(y=i\\) 的概率

- 定义指示函数：

	 $$ I(c)=\\left\\{ \begin{matrix} 1, \quad if \ c \ is \ true \\\\ 0, \quad if \ c \ is \ false \end{matrix} \\right. $$

- 对于多项式分布，\\(b(y)=1, \ \eta=\\left\[ \begin{matrix} log \frac{\phi\_{1}}{\phi\_{K}} \\\\ log \frac{\phi\_{2}}{\phi\_{K}} \\\\ ... \\\\ log \frac{\phi\_{K-1}}{\phi\_{K}} \end{matrix} \\right\], \ T(y)=\\left\[ \begin{matrix} I(y=1) \\\\ I(y=2) \\\\ ... \\\\ I(y=K-1) \end{matrix} \\right\], \ a(\eta)=log \ \left( 1 + \sum\_{i=1}^{K-1}e^{\eta\_{i}} \right)\\)

	$$
	\begin{align\*}
	p(y;\phi) &= \prod\_{i=1}^{K} \phi\_{i}^{I(y=i)} \newline
	&= exp \left( log \ \prod\_{i=1}^{K} \phi\_{i}^{I(y=i)} \right) \newline
	&= exp \left( \sum\_{i=1}^{K} {I(y=i)} \cdot log \ \phi\_{i} \right) \newline
	&= exp \left( \sum\_{i=1}^{K-1} \left( {I(y=i)} \cdot log \ \phi\_{i} \right) + \left( 1-\sum\_{i=1}^{K-1}{I(y=i)} \right) \cdot log \ \phi\_{K} \right) \newline
	&= exp \left( \sum\_{i=1}^{K-1} \left( {I(y=i)} \cdot log \frac{\phi\_{i}}{\phi\_{K}} \right) + log \ \phi\_{K} \right) \newline
	\end{align\*}
	$$

	- 由 \\(\eta=\\left\[ \begin{matrix} log \frac{\phi\_{1}}{\phi\_{K}} \\\\ log \frac{\phi\_{2}}{\phi\_{K}} \\\\ ... \\\\ log \frac{\phi\_{K-1}}{\phi\_{K}} \end{matrix} \\right\]\\) 可推导出：
	
		$$ \sum\_{i=1}^{K-1}e^{\eta\_{i}} = \frac{1-\phi\_{K}}{\phi\_{k}} \quad \Rightarrow \quad \phi\_{K} = \frac{1}{1+\sum\_{i=1}^{K-1}e^{\eta\_{i}}} $$

	- 将 \\(\phi\_{K}\\) 代入 \\(log \ \phi\_{K}\\) 可得：

		$$ p(y;\phi) = exp \left( \sum\_{i=1}^{K-1} \left( {I(y=i)} \cdot log \frac{\phi\_{i}}{\phi\_{K}} \right) - log \ \left( 1+\sum\_{i=1}^{K-1}e^{\eta\_{i}} \right) \right) $$

## 广义线性模型