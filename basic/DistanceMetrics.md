<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 距离度量

## 闵可夫斯基距离

- \\(N\\) 维空间中，两个样本 \\(x, \ y\\) 间的闵可夫距离定义为：

	$$ D(x,y) = \left( \sum\_{i=1}^{N} |x\_{i} - y\_{i}|^{p} \right) ^{\frac{1}{p}} $$

- \\(p=1\\) 时，得到曼哈顿距离：

	$$ D(x,y) = \sum\_{i=1}^{N} |x\_{i} - y\_{i}| $$

- \\(p=2\\) 时，得到欧几里德距离：

	$$ D(x,y) = \sqrt{\sum\_{i=1}^{N} (x\_{i} - y\_{i})^{2}} $$

- \\(p \rightarrow \infty\\) 时，得到切比雪夫距离：

	$$ D(x,y) = \max\_{i=1}^{N} |x\_{i} - y\_{i}| $$

## 交叉熵

- 参考 [Entropy.md] (Entropy.md)

## \\(KL\\) 散度

- Kullback-Leibler 散度

- 参考 [Entropy.md] (Entropy.md)

## \\(JS\\) 散度

- Jenssen-Shannon 散度

- \\(JS(p,q) = \frac{1}{2} KL(p||\frac{p+q}{2}) + \frac{1}{2} KL(q||\frac{p+q}{2})\\)，其中 \\(p\\) 是样本的真实分布，\\(q\\) 是样本的假设分布

- 与 \\(KL\\) 相比，\\(JS\\) 散度具有对称性：\\(JS(p,q) = JS(q,p)\\)

- 当真实分布 \\(p\\) 与假设分布 \\(q\\) 不重叠时，\\(JS\\) 散度为常数 \\(\log2\\)：

	$$
	\begin{aligned}
	prior \quad &p\_{i}=0 \quad when \quad q\_{i} \geq 0, \qquad q\_{i}=0 \quad when \quad p\_{i} \geq 0 \newline \newline
	\qquad JS(p,q) &= \frac{1}{2} \sum\_{i} p\_{i}\log\frac{2 p\_{i}}{p\_{i}+q\_{i}} + \frac{1}{2} \sum\_{i} q\_{i}\log\frac{2 q\_{i}}{p\_{i}+q\_{i}} \newline
	&= \frac{1}{2} \log2\ \sum\_{i} p\_{i} + \frac{1}{2} \sum\_{i} p\_{i}\log\frac{p\_{i}}{p\_{i}+q\_{i}} + \frac{1}{2} \log2\ \sum\_{i} q\_{i} + \frac{1}{2} \sum\_{i} q\_{i}\log\frac{q\_{i}}{p\_{i}+q\_{i}} \newline
	&= \log2 + \frac{1}{2} \sum\_{i} p\_{i}\log\frac{p\_{i}}{p\_{i}+q\_{i}} + \frac{1}{2} \sum\_{i} q\_{i}\log\frac{q\_{i}}{p\_{i}+q\_{i}} \qquad (1) \newline
	&= \log2 + \frac{1}{2} \sum\_{p\_{i} \neq 0} p\_{i}\log\frac{p\_{i}}{p\_{i}} + \frac{1}{2} \sum\_{q\_{i} \neq 0} q\_{i}\log\frac{q\_{i}}{q\_{i}} \qquad (2) \newline
	&= \log2
	\end{aligned}
	$$

- 由 \\((1)\\) 到 \\((2)\\) 的推导，由洛必达法则可得：

	$$
	\\left\\{ \begin{matrix} p\_{i}\log\frac{p\_{i}}{p\_{i}+q\_{i}} = 0 & if \ p\_{i} = 0, \ q\_{i} \neq 0 \\\\ q\_{i}\log\frac{q\_{i}}{p\_{i}+q\_{i}} = 0 & if \ q\_{i} = 0, \ p\_{i} \neq 0 \end{matrix} \\right\.
	$$

## Wasserstein 距离

- 又称 EM 距离，即 Earth-Mover 距离

- \\(W(p,q) = \inf \limits\_{\gamma \sim \Pi(p,q)} \mathbb{E}\_{(x,y) \sim \gamma}{\left[||x-y||\right]}\\)，其中 \\(\Pi(p,q)\\) 是 \\(p\\) 和 \\(q\\) 所有可能的联合分布的集合，\\((x,y)\\) 是从联合分布 \\(\gamma\\) 中采样的样本，\\(inf\\) 表示最大下界

- \\(\mathbb{E}\_{(x,y) \sim \gamma}{\left[||x-y||\right]}\\) 可以理解为在 \\(\gamma\\) 这个“路径规划”下，把 \\(q\\) 这堆“土”挪到 \\(p\\) “位置”所需的“消耗”，而 \\(W(p,q)\\) 就是在“最优路径规划”下的的“最小消耗”

- 与 \\(JS\\) 散度相比，即使没有重叠，Wasserstein 距离也能够反映两个分布的距离：

	- 考虑二维平面内两个均匀分布 \\(P\_{1}, P\_{2}\\):

		![img](images/wasserstein.png)
		
		- \\(JS(P\_{1}||P\_{2}) = \\left\\{ \begin{matrix} \log2 & if \ \theta \neq 0 \\\\ 0 & if \ \theta = 0 \end{matrix} \\right\.\\)

		- \\(W(P\_{1}, P\_{2}) = |\theta|\\)