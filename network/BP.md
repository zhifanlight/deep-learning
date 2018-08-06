<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Back Propagation

## 背景介绍

- BP 网络由一个输入层、一个输出层、一个或多个隐含层组成；每一层的输出使用 \\(sigmoid\\) 函数

- 正向计算时，从输入层开始，逐层计算每一层节点的输出；当前层的输出作为下一层的输入

- 反向传播时，从输出层开始，逐层计算每一层节点的误差；当前层的输出作为上一层的输入

- 在训练时，先正向计算，再反向传播，直到神经网络的权值趋于稳定

![img](images/bp.png)

## 数学推导

### 单样本

- 定义如下：

	- \\(t\_{i} \ \ \\)：样本 \\(s\\) 的实际值的第 \\(i\\) 个分量
	
	- \\(y\_{i} \ \ \\)：样本 \\(s\\) 的预测值的第 \\(i\\) 个分量

	- \\(w\_{ji}^{l} \ \ \\)：第 \\(l\\) 层第 \\(j\\) 个节点与第 \\(l-1\\) 层第 \\(i\\) 个节点间的权值

	- \\(b\_{j}^{l} \ \ \\)：第 \\(l\\) 层第 \\(j\\) 个节点的偏置

	- \\(x\_{j}^{l} \ \ \\)：第 \\(l\\) 层第 \\(j\\) 个节点的加权输入

		$$ x\_{j}^{l} = \sum\_{i}w\_{ji}^{l} \cdot z\_{i}^{l-1} + b\_{j}^{l} $$
	
	- \\(z\_{j}^{l} \ \ \\)：第 \\(l\\) 层第 \\(j\\) 个节点的预测输出

		$$ z\_{j}^{l} = f(x\_{j}^{l}) $$	

	- \\(E\_{s} \ \ \\)：输出层误差

		$$ E\_{s} = \frac{1}{2}\sum\_{i=1}^{N} (y\_{i} - t\_{i})^{2} $$

- 由梯度下降可得：

	$$
	\\left\\{ \begin{matrix}
	w\_{ji}^{l} \leftarrow w\_{ji}^{l} - \eta \cdot \frac {\partial{E\_{s}}}{\partial{w\_{ji}^{l}}} \\\\
	b\_{j}^{l} \leftarrow b\_{j}^{l} - \eta \cdot \frac {\partial{E\_{s}}}{\partial{b\_{j}^{l}}}
	\end{matrix} \\right\.
	$$

- 对权重项应用链式法则：

	$$ \frac{\partial{E\_{s}}}{\partial{w\_{ji}^{l}}} = \frac{\partial{E\_{s}}}{\partial{x\_{j}^{l}}} \cdot \frac{\partial{x\_{j}^{l}}}{\partial{w\_{ji}^{l}}} = \frac{\partial{E\_{s}}}{\partial{x\_{j}^{l}}} \cdot z\_{i}^{l-1} $$

	- 对于输出层，\\(x\_{j}^{l}\\) 通过影响 \\(y\_{i}\\) 直接作用于 \\(E\_{s}\\)，因此：

		$$
		\begin{align\*}
		\frac{\partial{E\_{s}}}{\partial{x\_{j}^{l}}} &= \frac{\partial{E\_{s}}}{\partial{y\_{j}}} \cdot \frac{\partial{y\_{j}}}{\partial{x\_{j}^{l}}} \newline
		&= (y\_{j} - t\_{j}) \cdot (y\_{j})' \newline
		\end{align\*}
		$$
	
		- 令 \\(\delta\_{j}^{l} = \frac{\partial{E\_{s}}}{\partial{x\_{j}^{l}}}\\)，即误差项为预测误差对该节点输入的偏导值，则：

			$$ \delta\_{j}^{l} = (y\_{j})' \ (y\_{j} - t\_{j}) $$
		
		- 于是 \\(w\_{ji}^{l}\\) 更新公式为：

			$$ w\_{ji}^{l} \leftarrow w\_{ji}^{l} - \eta \ \delta\_{j}^{l} \ z\_{i}^{l-1} $$
	
	- 对于中间层，\\(x\_{j}^{l}\\) 依次影响 \\(z\_{j}^{l}, x\_{k}^{l+1}, y\_{k}\\) 间接作用于 \\(E\_{s}\\)，因此：

		$$
		\begin{aligned}
		\frac{\partial{E\_{s}}}{\partial{x\_{j}^{l}}} &= \sum\_{k \in Next} \frac{\partial{E\_{s}}}{\partial{x\_{k}^{l+1}}} \cdot \frac{\partial{x\_{k}^{l+1}}}{\partial{z\_{j}^{l}}} \cdot \frac{\partial{z\_{j}^{l}}}{\partial{x\_{j}^{l}}} \newline
		&= \sum\_{k \in Next} \delta\_{k}^{l+1} \cdot w\_{kj}^{l+1} \cdot \left(z\_{j}^{l}\right)' \newline
		&= \left(z\_{j}^{l}\right)' \sum\_{k \in Next} \delta\_{k}^{l+1} w\_{kj}^{l+1} \newline
		\end{aligned}
		$$
	
		- 将 \\(\delta\_{j}^{l} = \frac{\partial{E\_{s}}}{\partial{a\_{j}^{l}}}\\) 代入得：

			$$ \delta\_{j}^{l} = \left(z\_{j}^{l}\right)' \sum\_{k \in Next} \delta\_{k}^{l+1} w\_{kj}^{l+1} $$

		- 于是 \\(w\_{ji}^{l}\\) 更新公式为：

			$$ w\_{ji}^{l} \leftarrow w\_{ji}^{l} - \eta \ \delta\_{j}^{l} \ z\_{i}^{l-1} $$

- 对偏置项应用链式法则：

	$$ \frac{\partial{E\_{s}}}{\partial{b\_{j}^{l}}} = \frac{\partial{E\_{s}}}{\partial{x\_{j}^{l}}} \cdot \frac{\partial{x\_{j}^{l}}}{\partial{b\_{j}^{l}}} = \frac{\partial{E\_{s}}}{\partial{x\_{j}^{l}}} $$
	
	- 于是：

		$$ b\_{j}^{l} \leftarrow b\_{j}^{l} - \eta \ \delta\_{j}^{l} $$

### 样本集

- 定义如下：

	- \\(\lambda \ \ \\)：\\(L\_{2}\\) 正则项系数

	- \\(J \ \ \\)：样本集上的损失函数

		$$ J = \frac{1}{m} \sum\_{s=1}^{m} E\_{s} + \frac{\lambda}{2} \sum\_{l} \sum\_{j} \sum\_{i} (w\_{ji}^{l})^{2} $$

- 对权重项应用链式法则：

	$$ \frac{\partial{J}}{\partial{w\_{ji}^{l}}} = \frac{1}{m} \sum\_{s=1}^{m} \frac{\partial{E\_{s}}}{\partial{w\_{ji}^{l}}} + \lambda w\_{ji}^{l} $$

- 对偏置项应用链式法则：

	$$ \frac{\partial{J}}{\partial{b\_{j}^{l}}} = \frac{1}{m} \sum\_{s=1}^{m} \frac{\partial{E\_{s}}}{\partial{b\_{j}^{l}}} $$