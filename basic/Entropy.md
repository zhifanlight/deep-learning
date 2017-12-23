<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 熵

## 信息熵（自信息）

- 描述一个随机变量所需要的平均信息量

- 假设 \\(p\\) 是样本的真实分布，则熵计算如下：

	$$ H(p) = -\sum_{i} p(i) \cdot log \ p(i) $$

	- 底数是 \\(2\\) 时，单位是比特

	- 底数是 \\(e\\) 时，单位是奈特

- 熵的取值范围是 \\([0, logK]\\)，其中 \\(K\\) 是样本的取值个数

	- 当只有一项 \\(p(i)\\) 取值为 \\(1\\)，而其他 \\(p(i)\\) 取值均为 \\(0\\) 时，\\(H(p) = 0\\)

	- 当 \\(p(i)\\) 均匀分布，即 \\(p(1) = p(2) = \ ... \ = p(K)\\) 时，\\(H(p) = logK\\)

## 联合熵

- 描述一对随机变量所需要的平均信息量

- 假设 \\(X, Y\\) 是一对离散型随机变量，\\((X,Y) \sim p(x,y)\\)，则联合熵计算如下：

	$$ H(X,Y) = -\sum\_{x \in X} \sum\_{y \in Y} p(x,y) \cdot log \ p(x,y) $$

- 推导过程：

	$$
	\begin{align\*}
	H(X,Y) &= \sum\_{x \in X} p(x) \cdot H(x, Y) \newline
	&= \sum\_{x \in X} p(x) \cdot \left[ -\sum\_{y \in Y|x} p(y) \cdot log \ p(x,y) \right] \newline
	&= -\sum\_{x \in X} \sum\_{y \in Y|x} p(x,y) \cdot log \ p(x,y)
	\end{align\*}
	$$

## 条件熵

- 对于联合分布 \\((X,Y)\\)，在已知 \\(Y\\) 时，描述 \\(X\\) 所需要的平均信息量，计算如下：

	$$ H(X|Y) = -\sum\_{x \in X} \sum\_{y \in Y} p(x,y) \cdot log \ p(x|y) $$

- 推导过程：

	$$
	\begin{align\*}
	H(X|Y) &= \sum\_{y \in Y} p(y) \cdot H(X|y) \newline
	&= \sum\_{y \in Y} p(y) \cdot \left[ -\sum\_{x \in X|y} p(x) \cdot log \ p(x|y) \right] \newline
	&= -\sum\_{y \in Y} \sum\_{x \in X|y} p(x,y) \cdot log \ p(x|y) \newline
	&= -\sum\_{y \in Y} \sum\_{x \in X|y} p(x,y) \cdot log \ \frac{p(x,y)}{p(y)}
	\end{align\*}
	$$

- 条件熵是联合熵、信息熵之差：

	$$ H(X|Y) = H(X,Y) - H(Y) $$

## 互信息

- 对于联合分布 \\((X,Y)\\)，描述 \\(X, Y\\) 的依赖程度，计算如下：
	
	$$ I(X;Y) = \sum\_{x \in X} \sum\_{y \in Y} p(x,y) \cdot log \frac{p(x,y)}{p(x) \cdot p(y)} $$

- 上式为广义 \\(KL\\) 散度，\\(KL = 0\\) 当且仅当  \\(p(x,y) = p(x) \cdot p(y)\\)，即 \\(X,Y\\) 相互独立

- 在已知 \\(Y\\) 时，描述 \\(X\\) 节省的平均信息量，即 \\(I(X;Y) = H(X) - H(X|Y)\\)。证明：

	$$
	\begin{align\*}
	I(X;Y) &= \sum\_{x \in X} \sum\_{y \in Y|x} p(x,y) \cdot log \frac{p(x,y)}{p(x) \cdot p(y)} \newline
	&= \sum\_{x \in X} \sum\_{y \in Y|x} p(x,y) \cdot log \frac{p(x,y)}{p(y)} - \sum\_{x \in X} \sum\_{y \in Y|x} p(x,y) \cdot log \ p(x) \newline
	&= -H(X|Y) - \sum\_{x \in X} p(x) \cdot log \ p(x) \newline
	&= H(X) - H(X|Y) \newline
	\end{align\*}
	$$
	
## 信息增益

- 与互信息相比，计算方式相同，意义不同：

	- 在计算互信息时，\\(Y\\) 是普通的随机事件

	- 在计算信息增益时，\\(Y\\) 是 \\(X\\) 的分类方式

## 相对熵（\\(KL\\) 散度）

- Kullback-Leibler 散度

- 衡量假设分布 \\(q\\) 与真实分布 \\(p\\) 之间的差距，计算如下：

	$$ KL(p||q) = \sum\_{i} p(i) \cdot log \frac{p(i)}{q(i)} $$

- \\(KL\\) 散度不满足对称性：\\(KL(p||q) \neq KL(q||p)\\)

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

- 最小化 \\(KL\\) 散度等价于最大化对数似然函数。证明：

	- 用 \\(q(x;\theta)\\) 来近似真实分布 \\(p(x)\\)，已知来自 \\(p(x)\\) 的样本集 \\(S = {x\_{1}, x\_{2}, ..., x\_{N}}\\)：

		$$
		\begin{align\*}
		min \ KL(p||q) &= \sum\_{i} p(i) \cdot log \ \frac{p(i)}{q(i)} \newline
		&= \sum\_{i} p(i) \cdot (log \ p(i) - log \ q(i)) \newline
		&= \sum\_{x \in S} \frac{1}{N} \cdot \left(log \ \frac{1}{N}  - log \ q(x;\theta) \right) \newline \newline
		min \ KL(p||q) &= max \ \sum\_{x \in S} log\ q(x;\theta)
		\end{align\*}
		$$

## 交叉熵

- 用假设分布 \\(q\\) 描述真实分布 \\(p\\) 所需要的平均信息量

- 衡量假设分布 \\(q\\) 与真实分布 \\(p\\) 之间的差距，计算如下：

	$$ H(p,q) = -\sum\_{i}p(i) \cdot log \ q(i) $$
	
- 在真实分布 \\(p\\) 已知的情况下，交叉熵与相对熵在行为上等价：都反应真实分布 \\(p\\) 和假设分布 \\(q\\) 的差距

- 交叉熵是熵、相对熵之和：

	$$ H(p,q) = H(p) + KL(p||q) $$

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

	$$p\_{i} = \frac{exp(x\_{i})}{\sum\_{j}exp(x\_{j})}$$

- 计算真实类别与预测输出的交叉熵：

	$$H = -\sum\_{i}y\_{i} \cdot log \ p\_{i}$$