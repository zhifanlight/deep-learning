<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 熵

## 信息熵

- 衡量编码一个样本所需要的平均 bit 数

- \\(H(p) = -\sum_{i} p(i) \cdot log \ p(i)\\)，其中 \\(p\\) 是样本的真实分布

## 相对熵（\\(KL\\) 散度）

- Kullback-Leibler 散度

- 表示在真实分布为 \\(p\\) 的前提下，使用 \\(q\\) 分布进行编码相对于使用真实分布 \\(p\\) 进行编码所多出来的 bit 数

- 当真实分布为 \\(p\\) 时，假设分布 \\(q\\) 的无效性，用来衡量分布 \\(p,q\\) 的距离

- \\(KL(p||q) = \sum_{i} p(i) \cdot log \frac{p(i)}{q(i)}\\)，其中 \\(p\\) 是样本的真实分布，\\(q\\) 是样本的假设分布

- \\(KL\\) 散度不具有对称性：\\(KL(p||q) \neq KL(q||p)\\)

- \\(KL(p||q) \geq 0\\) 恒成立，当且仅当 \\(p=q\\) 时等号成立。证明：

	$$
	\begin{align\*}
	prior \qquad &lnx \leq (x-1) \qquad \forall x \newline \newline
	set \qquad f &= -KL(p||q) \newline
	&= \sum\_{i} p(i) \cdot log \frac{q(i)}{p(i)} \newline
	&\leq \sum\_{i} p(i) \cdot (\frac{q(i)}{p(i)} - 1) \newline
	&= \sum\_{i} (q(i) - p(i)) \newline
	&= 0 \newline \newline
	\Rightarrow \qquad &KL(p||q) \geq 0
	\end{align\*}
	$$

## 交叉熵

- 表示在真实分布为 \\(p\\) 的前提下，使用 \\(q\\) 分布进行编码所需要的平均 bit 数

- 在真实分布 \\(p\\) 已知的情况下，交叉熵与相对熵在行为上等价：都反应真实分布 \\(p\\) 和假设分布 \\(q\\) 的相似性

- \\(H(p,q) = -\sum_{i}p(i) \cdot log \ q(i)\\)，其中 \\(p\\) 是样本的真实分布，\\(q\\) 是样本的假设分布

- 由定义，交叉熵为熵与相对熵之和：\\(H(p,q) = H(p) + KL(p||q)\\)

### Sigmoid 交叉熵

$$
\begin{align\*}
H &= -y \cdot log \ p - (1 - y) \cdot log \ (1 - p) \newline
&= -y \cdot log \ \left( \frac{1}{1 + exp(-x)} \right) - (1 - y) \cdot log \ \left( 1 - \frac{1}{1 + exp(-x)} \right) \newline
&= -y \cdot log \ \left( \frac{1}{1 + exp(-x)} \right) - (1 - y) \cdot log \ \left(\frac{exp(-x)}{1 + exp(-x)} \right) \newline
&= -y \cdot log \ \left( \frac{1}{1 + exp(-x)} \right) - (1 - y) \cdot \left(-x + log \ \left(\frac{1}{1 + exp(-x)}\right) \right) \newline
&= (1 - y) \cdot x - log \ \left(\frac{1}{1 + exp(-x)}\right) \newline
&= x - xy + log \ (1 + exp(-x)) \newline
\end{align\*}
$$

- 当 \\(x < 0\\) 时，为避免 \\(exp(-x)\\) 溢出，可对上式进行变换：

	$$
	\begin{align\*}
	H &= x - xy + log \ (1 + exp(-x)) \newline
	&= -xy + log \ exp(x) + log \ (1 + exp(-x)) \newline
	&= -xy + log \ (exp(x) + 1) \newline
	\end{align\*}
	$$
	
- 综上所述，Sigmoid 交叉熵计算如下：

	$$H = max(x, 0) - xy + log(1 + exp(-|x|))$$

### Softmax 交叉熵

- 首先计算预测输出的 Softmax 值：

	$$p\_{i} = \frac{exp(o\_{i})}{\sum\_{j}exp(o\_{j})}$$

- 计算真实类别与预测输出的交叉熵：

	$$H = -\sum\_{i}y\_{i} \cdot log \ p\_{i}$$