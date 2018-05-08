<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 朴素贝叶斯

## 基本思想

- 假设输入向量各个维度之间相互独立

- 通过学习联合分布 \\(P(X,Y)\\) 来学习条件概率 \\(P(Y|X)\\)

## 数学推导

### 基本方法

- 假设输入为 \\(N\\) 维向量 \\(x\\)，输出为标量 \\(y \in \\{c\_{1}, \ c\_{2}, \ \cdots, c\_{K}\\}\\)

- 条件概率计算如下：

	$$ P(X=x | Y=c\_{k}) = P(X\_{1}=x\_{1}, \ X\_{2}=x\_{2}, \ \cdots \ X\_{N}=x\_{N} | Y=c\_{k}) $$

	- 假设 \\(x\_{j}\\) 的取值有 \\(S\_{j}\\) 个，模型的参数个数为：

		$$ K \prod\_{j=1}^{N} S\_{j} $$

- 如果假设输入的各个维度间相互独立，条件概率计算如下：

	$$ P(X=x | Y=c\_{k}) = \prod\_{j=1}^{N} P(X\_{j}=x\_{j} | Y=c\_{k})$$

	- 模型的参数个数为：

		$$ K \sum\_{j=1}^{N} S\_{j} $$

- 朴素贝叶斯后验概率计算如下：

	$$ P(Y=c\_{k} | X=x) = \frac{P(X=x, Y=c\_{k})}{P(X=x)} $$

	- 当 \\(x\\) 固定时，\\(P(X=x)\\) 为常量；因此后验概率计算如下：

		$$
		\begin{aligned}
		P(Y=c\_{k} | X=x) &\propto P(X=x, Y=c\_{k}) \newline
		&= P(Y=c\_{k}) \cdot P(X=x | Y=c\_{k}) \newline
		&= P(Y=c\_{k}) \prod\_{j=1}^{N} P(X\_{j}=x\_{j} | Y=c\_{k}) \newline
		\end{aligned}
		$$

	- 选择最大 \\(P(Y=c\_{k} | X=x)\\) 对应的 \\(c\_{k}\\) 作为最终类别标签

### 参数估计

- 先验概率：

	$$ P(Y=c\_{k}) = \frac{\sum\_{i=1}^{M} I(y\_{i} = c\_{k})}{M} $$

- 条件概率：

	$$ P(X\_{j}=a\_{t} | Y=c\_{k}) = \frac{\sum\_{i=1}^{M} I(x\_{ij}=a\_{t}, y\_{i}=c\_{k})}{\sum\_{i=1}^{M}(y\_{i}=c\_{k})} $$

### 拉普拉斯平滑

- 采用朴素贝叶斯估计时，某些概率值可能为 \\(0\\)

- 先验概率计算如下：

	$$ P(Y=c\_{k}) = \frac{\lambda + \sum\_{i=1}^{M} I(y\_{i} = c\_{k})}{K \lambda + M} $$

- 条件概率计算如下：

	$$ P(X\_{j}=a\_{t} | Y=c\_{k}) = \frac{\lambda + \sum\_{i=1}^{M} I(x\_{ij}=a\_{t}, y\_{i}=c\_{k})}{|S\_{j}| \cdot \lambda + \sum\_{i=1}^{M}(y\_{i}=c\_{k})} $$

	- 其中，\\(S\_{j}\\) 是 \\(x\_{j}\\) 的取值集合

- \\(\lambda\\) 通常为 \\(1\\)