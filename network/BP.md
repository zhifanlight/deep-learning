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

	- \\(w\_{ji} \ \ \\)：当前层第 \\(j\\) 个节点与上一层第 \\(i\\) 个节点间的权值

	- \\(b\_{j} \ \ \\)：当前层第 \\(j\\) 个节点的偏置

	- \\(x\_{i} \ \ \\)：上一层第 \\(i\\) 个节点的输出

	- \\(net\_{j} \ \ \\)：当前层第 \\(j\\) 个节点的加权输入

		$$ net\_{j} = \sum\_{i}w\_{ji} \cdot x\_{i} + b\_{j} $$
	
	- \\(z\_{j} \ \ \\)：当前层第 \\(j\\) 个节点的预测输出

		$$
		\\left\\{ \begin{matrix}
		z\_{j} = sigmoid(net\_{j}) = \frac{1}{1 + e^{-net\_{j}}} \\\\
		\frac{\partial{z\_{j}}}{\partial{net\_{j}}} = z\_{j} (1 - z\_{j})
		\end{matrix} \\right\.
		$$	

	- \\(E\_{s} \ \ \\)：输出层误差

		$$ E\_{s} = \frac{1}{2}\sum\_{i=1}^{N} (t\_{i} - y\_{i})^{2} $$

- 由梯度下降可得：

	$$
	\\left\\{ \begin{matrix}
	w\_{ji} \leftarrow w\_{ji} - \eta \cdot \frac {\partial{E\_{s}}}{\partial{w\_{ji}}} \\\\
	b\_{j} \leftarrow b\_{j} - \eta \cdot \frac {\partial{E\_{s}}}{\partial{b\_{j}}}
	\end{matrix} \\right\.
	$$

- 对权重项应用链式法则：

	$$ \frac{\partial{E\_{s}}}{\partial{w\_{ji}}} = \frac{\partial{E\_{s}}}{\partial{net\_{j}}} \cdot \frac{\partial{net\_{j}}}{\partial{w\_{ji}}} = \frac{\partial{E\_{s}}}{\partial{net\_{j}}} \cdot x\_{ji} $$

	- 对于输出层，\\(net\_{j}\\) 通过影响 \\(y\_{i}\\) 直接作用于 \\(E\_{d}\\)，因此：

		$$
		\begin{align\*}
		\frac{\partial{E\_{s}}}{\partial{net\_{j}}} &= \frac{\partial{E\_{s}}}{\partial{y\_{j}}} \cdot \frac{\partial{y\_{j}}}{\partial{net\_{j}}} \newline
		&= [-(t\_{j} - y\_{j})] \cdot [y\_{j} (1 - y\_{j})] \newline
		&= -(t\_{j} - y\_{j}) y\_{j} (1 - y\_{j}) \newline
		\end{align\*}
		$$
	
		- 令 \\(\delta\_{j} = -\frac{\partial{E\_{s}}}{\partial{net\_{j}}}\\)，即误差项为预测误差对该节点输入的偏导值的相反数，则：

			$$ \delta\_{j} = y\_{j} (1 - y\_{j}) (t\_{j} - y\_{j}) $$
		
		- 于是 \\(w\_{ji}\\) 更新公式为：

			$$ w\_{ji} \leftarrow w\_{ji} + \eta\delta\_{j}x\_{ji} $$
	
	- 对于中间层，\\(net\_{j}\\) 依次影响 \\(z\_{j}, net\_{k}, y\_{k}\\) 间接作用于 \\(E\_{d}\\)，因此：

		$$
		\begin{align\*}
		\frac{\partial{E\_{s}}}{\partial{net\_{j}}} &= \sum\_{k \in Next} \frac{\partial{E\_{s}}}{\partial{net\_{k}}} \cdot \frac{\partial{net\_{k}}}{\partial{z\_{j}}} \cdot \frac{\partial{z\_{j}}}{\partial{net\_{j}}} \newline
		&= \sum\_{k \in Next} (-\delta\_{k}) \cdot w\_{kj} \cdot [z\_{j}(1 - z\_{j})] \newline
		&= -z\_{j}(1 - z\_{j}) \sum\_{k \in Next} \delta\_{k}w\_{kj} \newline
		\end{align\*}
		$$
	
		- 将 \\(\delta\_{j} = -\frac{\partial{E\_{s}}}{\partial{net\_{j}}}\\) 代入得：

			$$ \delta\_{j} = z\_{j}(1 - z\_{j}) \sum\_{k \in Next} \delta\_{k} w\_{kj} $$

		- 于是 \\(w\_{ji}\\) 更新公式为：

			$$ w\_{ji} \leftarrow w\_{ji} + \eta\delta\_{j}x\_{ji} $$

- 对偏置项应用链式法则：

	$$ \frac{\partial{E\_{s}}}{\partial{b\_{j}}} = \frac{\partial{E\_{s}}}{\partial{net\_{j}}} \cdot \frac{\partial{net\_{j}}}{\partial{b\_{j}}} = \frac{\partial{E\_{s}}}{\partial{net\_{j}}} $$
	
	- 于是：

		$$ b\_{j} \leftarrow b\_{j} + \eta\delta\_{k}$$

### 样本集

- 定义如下：

	- \\(w\_{ji}^{(k)} \ \ \\)：第 \\(k\\) 层第 \\(j\\) 个节点与第 \\(k-1\\) 层第 \\(i\\) 个节点间的权值

	- \\(b\_{j}^{(k)} \ \ \\)：第 \\(k\\) 层第 \\(j\\) 个节点的偏置

	- \\(\lambda \ \ \\)：\\(L\_{2}\\) 正则项的系数

	- \\(J \ \ \\)：样本集上的损失函数

		$$ J = \frac{1}{m} \sum\_{s=1}^{m} E\_{s} + \frac{\lambda}{2} \sum\_{k} \sum\_{j} \sum\_{i} (w\_{ji}^{(k)})^{2} $$

- 对权重项应用链式法则：

	$$ \frac{\partial{J}}{\partial{w\_{ji}^{(l)}}} = \frac{1}{m} \sum\_{s=1}^{m} \frac{\partial{E\_{s}}}{\partial{w\_{ji}^{(l)}}} + \lambda w\_{ji}^{(l)} $$

- 对偏置项应用链式法则：

	$$ \frac{\partial{J}}{\partial{b\_{j}^{(l)}}} = \frac{1}{m} \sum\_{s=1}^{m} \frac{\partial{E\_{s}}}{\partial{b\_{j}^{(l)}}} $$