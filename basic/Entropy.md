<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 熵

## 熵（自信息）

- 描述一个随机变量所需要的平均信息量

- 假设 \\(p\\) 是样本的真实分布，则熵计算如下：

	$$ H(p) = -\sum_{i} p\_{i} \cdot \log \ p\_{i} $$

	- 底数是 \\(2\\) 时，单位是比特

	- 底数是 \\(e\\) 时，单位是奈特

- 熵的取值范围是 \\([0, \log K]\\)，其中 \\(K\\) 是样本的取值个数

	- 当只有一项 \\(p\_{i}\\) 取值为 \\(1\\)，而其他 \\(p\_{i}\\) 取值均为 \\(0\\) 时，\\(H(p) = 0\\)

	- 当 \\(p\_{i}\\) 均匀分布，即 \\(p\_{1} = p\_{2} = \ \cdots \ = p\_{K}\\) 时，\\(H(p) = \log K\\)

		- 优化目标：

			$$
			f(p\_{1}, \ p\_{2}, \ \cdots, \ p\_{K}) = -\sum\_{k=1}^{K} p\_{k} \cdot \log p\_{k}
			$$
		
		- 约束条件：

			$$
			\sum\_{k=1}^{K} p\_{k} = 1
			$$
		
		- 由拉格朗日乘子法：

			$$ L(p, \lambda) = -\sum\_{k=1}^{K} p\_{k} \cdot \log p\_{k} + \lambda \left( \sum\_{k=1}^{K} p\_{k} - 1 \right) $$
		
			- 计算偏导：

				$$ \frac{\partial{L}}{\partial{p\_{k}}} = - \left( \frac{1}{\ln2} + \log p\_{k} \right) - \lambda $$
				
				$$ \frac{\partial{L}}{\partial{\lambda}} = \sum\_{k=1}^{K} p\_{k} - 1 $$
				
			- 令偏导为 \\(0\\)：

				$$ p\_{1} = p\_{2} = \cdots = p\_{k} = \frac{1}{K} $$
			
			- 代入信息熵计算公式可得：

				$$ H(p) = \log K $$

## 相对熵（\\(KL\\) 散度）

- Kullback-Leibler 散度

- 衡量假设分布 \\(q\\) 与真实分布 \\(p\\) 之间的差距，计算如下：

	$$ KL(p||q) = \sum\_{i} p\_{i} \cdot \log \frac{p\_{i}}{q\_{i}} $$

- \\(KL\\) 散度不满足对称性：\\(KL(p||q) \neq KL(q||p)\\)

- \\(KL(p||q) \geq 0\\) 恒成立，当且仅当 \\(p=q\\) 时等号成立。证明：

	$$
	\begin{aligned}
	prior \qquad &\ln x \leq (x-1) \qquad \forall x \newline \newline
	set \qquad f &= -KL(p||q) \newline
	&= \sum\_{i} p\_{i} \cdot \log \frac{q\_{i}}{p\_{i}} \newline
	&\leq \sum\_{i} p\_{i} \cdot (\frac{q\_{i}}{p\_{i}} - 1) \newline
	&= \sum\_{i} (q\_{i} - p\_{i}) \qquad iif \ \ p\_{i} = q\_{i} \newline
	&= 0 \newline \newline
	\Rightarrow \qquad &KL(p||q) \geq 0
	\end{aligned}
	$$

- 最小化 \\(KL\\) 散度等价于最大化对数似然。证明：

	- 用 \\(q(x;\theta)\\) 来近似真实分布 \\(p(x)\\)，已知来自 \\(p(x)\\) 的样本集 \\(S = {x\_{1}, x\_{2}, \cdots, x\_{N}}\\)：

		$$
		\begin{aligned}
		\min \ KL(p||q) &= \sum\_{i} p\_{i} \cdot \log \ \frac{p\_{i}}{q\_{i}} \newline
		&= \sum\_{i} p\_{i} \cdot (\log \ p\_{i} - \log \ q\_{i}) \newline
		&= \sum\_{x \in S} \frac{1}{N} \cdot \left(\log \ \frac{1}{N}  - \log \ q(x|\theta) \right) \newline \newline
		\min \ KL(p||q) &= \max \ \sum\_{x \in S} \log\ q(x|\theta)
		\end{aligned}
		$$

## 交叉熵

- 衡量假设分布 \\(q\\) 与真实分布 \\(p\\) 之间的差距，计算如下：

	$$ H(p,q) = -\sum\_{i}p\_{i} \cdot \log \ q\_{i} $$
	
- 在真实分布 \\(p\\) 已知的情况下，交叉熵与相对熵在行为上等价：都反应真实分布 \\(p\\) 和假设分布 \\(q\\) 的差距

- 交叉熵是熵与相对熵之和：

	$$ H(p,q) = H(p) + KL(p||q) $$

### Sigmoid 交叉熵

$$
\begin{aligned}
H &= -y \cdot \log \ p - (1 - y) \cdot \log \ (1 - p) \newline
&= -y \cdot \log \ \left( \frac{1}{1 + \exp(-x)} \right) - (1 - y) \cdot \log \ \left( 1 - \frac{1}{1 + \exp(-x)} \right) \newline
&= -y \cdot \log \ \left( \frac{1}{1 + \exp(-x)} \right) - (1 - y) \cdot \log \ \left(\frac{\exp(-x)}{1 + \exp(-x)} \right) \newline
&= -y \cdot \log \ \left( \frac{1}{1 + \exp(-x)} \right) - (1 - y) \cdot \left(-x + \log \ \left(\frac{1}{1 + \exp(-x)}\right) \right) \newline
&= (1 - y) \cdot x - \log \ \left(\frac{1}{1 + \exp(-x)}\right) \newline
&= x - xy + \log \ (1 + \exp(-x)) \newline
\end{aligned}
$$

- 当 \\(x < 0\\) 时，为避免 \\(\exp(-x)\\) 溢出，可对上式进行变换：

	$$
	\begin{aligned}
	H &= x - xy + \log \ (1 + \exp(-x)) \newline
	&= -xy + \log \ \exp(x) + \log \ (1 + \exp(-x)) \newline
	&= -xy + \log \ (\exp(x) + 1) \newline
	\end{aligned}
	$$
	
- 综上所述，Sigmoid 交叉熵计算如下：

	$$H = \max(x, 0) - xy + \log(1 + \exp(-|x|))$$

### Softmax 交叉熵

- 首先计算预测输出的 Softmax 值：

	$$p\_{i} = \frac{\exp(x\_{i})}{\sum\_{j}\exp(x\_{j})}$$

- 计算真实类别与预测输出的交叉熵：

	$$H = -\sum\_{i}y\_{i} \cdot \log \ p\_{i}$$