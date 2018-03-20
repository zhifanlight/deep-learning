<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 数学定理

## Jensen 不等式

- 向下凸起的函数称为凸函数，凸函数有最小值

- 对于凸函数上的任意两点，这两点的割线一定在这两点间函数图像的上方，即：

	$$ a f(x\_{1}) + b f(x\_{2}) \geq f(a x\_{1} + b x\_{2}) $$
	
	- 其中 \\(0 \leq a,b \leq 1\\) 且 \\(a + b = 1\\)

	![img](images/convex.png)

## 大数定律

- 对于 \\(n\\) 个独立同分布的样本，当 \\(n \rightarrow \infty\\) 时，样本均值收敛到期望值：

	$$ \lim\_{n \rightarrow \infty}{\frac{1}{n} \sum\_{i=1}^{n} x\_{i}} = \mu $$

## 中心极限定理

- 对于 \\(n\\) 个独立同分布的样本，当 \\(n \rightarrow \infty\\) 时，样本均值近似服从正态分布：

	$$ \lim\_{n \rightarrow \infty} \frac{1}{n} \sum\_{i=1}^{n} x\_{i} \ \sim \ N(\mu, \frac{\sigma^{2}}{n}) $$

## 泰勒展开

- 函数 \\(f(x)\\) 的在 \\(x=a\\) 处的泰勒展开如下：

	$$ f(x) = \sum\_{n=0}^{\infty} \frac{f^{n}(a)}{n!} (x-a)^{n} $$