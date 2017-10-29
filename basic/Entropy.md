<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 熵

&nbsp;

## 熵

- 衡量编码一个样本所需要的平均 bit 数

- \\(H(p) = \sum_{i} p(i) \cdot log \frac{1}{p(i)}\\)，其中 \\(p\\) 是样本的真实分布

## 相对熵（\\(KL\\)散度）

- Kullback-Leibler 散度

- 表示在真实分布为 \\(p\\) 的前提下，使用 \\(q\\) 分布进行编码相对于使用真实分布 \\(p\\) 进行编码所多出来的 bit 数

- 当真实分布为 \\(p\\) 时，假设分布 \\(q\\) 的无效性，用来衡量分布 \\(p,q\\) 的距离

- \\(D\_{KL}(p||q) = \sum_{i} p(i) \cdot log \frac{p(i)}{q(i)}\\)，其中 \\(p\\) 是样本的真实分布，\\(q\\) 是样本的假设分布

- \\(D_{KL} \geq 0\\) 恒成立，当且仅当 \\(p=q\\) 时等号成立。证明：

$$
\begin{align\*}
prior \qquad &lnx \leq (x-1) \qquad \forall x \newline \newline
set \qquad f &= -D_{KL} \newline
&= \sum\_{i} p(i) \cdot log \frac{q(i)}{p(i)} \newline
&\leq \sum\_{i} p(i) \cdot (\frac{q(i)}{p(i)} - 1) \newline
&= \sum\_{i} (q(i) - p(i)) \newline
&= 0 \newline \newline
\Rightarrow \qquad &D\_{KL} \geq 0
\end{align\*}
$$

## 交叉熵

- 表示在真实分布为 \\(p\\) 的前提下，使用 \\(q\\) 分布进行编码所需要的平均 bit 数

- 在真实分布 \\(p\\) 已知的情况下，交叉熵与相对熵在行为上等价：都反应真实分布 \\(p\\) 和假设分布 \\(q\\) 的相似性

- \\(H(p,q) = \sum_{i}p(i) \cdot log \frac {1} {q(i)}\\)，其中 \\(p\\) 是样本的真实分布，\\(q\\) 是样本的假设分布

- 由定义，交叉熵为熵与相对熵之和：\\(H(p,q) = H(p) + D_{KL}(p||q)\\)

## \\(JS\\)散度

- Jenssen-Shannon 散度

- \\(JS(p,q) = \frac{1}{2} D\_{KL}(p||\frac{p+q}{2}) + \frac{1}{2} D\_{KL}(q||\frac{p+q}{2})\\)

- 与 \\(KL\\) 相比，\\(JS\\) 散度具有对称性：\\(JS(p,q) = JS(q,p)\\)

- 当真实分布为 \\(p\\) 与假设分布 \\(q\\) 不重叠时，\\(JS\\) 散度为常数 \\(log2\\)：

$$
\begin{align\*}
prior \quad &p(i)=0 \quad when \quad q(i) \geq 0, \qquad q(i)=0 \quad when \quad p(i) \geq 0 \newline \newline
\qquad JS(p,q) &= \frac{1}{2} \sum\_{i} p(i)log\frac{p(i)}{\frac{p(i)+q(i)}{2}} + \frac{1}{2} \sum\_{i} q(i)log\frac{q(i)}{\frac{p(i)+q(i)}{2}} \newline
&= \frac{1}{2} log2\ \sum\_{i} p(i) + \frac{1}{2} \sum\_{i} p(i)log\frac{p(i)}{p(i)+q(i)} + \frac{1}{2} log2\ \sum\_{i} q(i) + \frac{1}{2} \sum\_{i} q(i)log\frac{q(i)}{p(i)+q(i)} \newline
&= log2 + \frac{1}{2} \sum\_{i} p(i)log\frac{p(i)}{p(i)+q(i)} + \frac{1}{2} \sum\_{i} q(i)log\frac{q(i)}{p(i)+q(i)} \newline
&= log2 + \frac{1}{2} \sum\_{i} p(i)log\frac{p(i)}{p(i)} + \frac{1}{2} \sum\_{i} q(i)log\frac{q(i)}{q(i)} \newline
&= log2
\end{align\*}
$$