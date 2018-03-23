<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 逻辑回归

## 背景介绍

- 在分类任务中，最简单的分类器是二元线性分类模型，其结果是一个超平面：

	- 超平面两侧的数据，具有相反的类别标签

	- 根据预测数据与超平面的位置关系，输出其类别标签

- 对于二分类任务，假设获得了 \\(N\\) 维空间中的 \\(M\\) 个点及观察值：\\((x\_{1},y\_{1}), (x\_{2},y\_{2}), \cdots, (x\_{M},y\_{M})\\)，其中 \\(x\_{i}\\) 是 \\(N\\) 维列向量 \\(\\left\[ \begin{matrix} x\_{i1} \\\\ x\_{i2} \\\\ \vdots \\\\ x\_{iN} \end{matrix} \\right\]\\)，\\(y\_{i}\\) 是标量

## 数学推导

### 模型建立

- 如果把 \\(N\\) 维列向量 \\(x\_{i}\\) 扩充到 \\(N + 1\\) 维并令 \\(x\_{i0}=1\\)，那么待拟合超平面可以定义为 \\(g(x) = \theta^{T}x\\)，其中 \\(\theta\\) 是 \\(N + 1\\) 维列向量

- 理想的决策边界与分类规则如下：

	$$ h(g(x)) = \\left\\{ \begin{matrix} 1, \ \ \quad g(x) > 0 \\\\ 0.5, \quad g(x) = 0 \\\\ 0, \ \ \quad g(x) < 0 \end{matrix} \\right. $$

	- 由于 \\(h\\) 要么不可导，要么导数为 \\(0\\)，无法使用数值方法优化求解

- 可以使用 sigmoid 函数代替 \\(h\\)：

	$$ h(g(x)) = \frac{1}{1 + e^{-g(x)}} $$
		
	- sigmoid 函数具有以下性质：

		$$ \\left\\{ \begin{matrix} g(x) > 0 \quad \rightarrow \quad h(g(x)) > 0.5 \\\\ g(x) = 0 \quad \rightarrow \quad h(g(x)) = 0.5 \\\\ g(x) < 0 \quad \rightarrow \quad h(g(x)) < 0.5 \end{matrix} \\right. $$
		
	- sigmoid 函数值域为 \\((0,1)\\)，因此 \\(h(g(x))\\) 表示 \\(x\\) 类别为 \\(1\\) 的概率，简记为 \\(h\_{\theta}(x)\\)

- 通过比较概率值 \\(P(y=1|x), \ P(y=0|x)\\) 进行类别判断

- 事件的几率表示事件发生概率与不发生概率的比值，对数几率计算如下：

	$$ logit(p) = log \frac{p}{1-p} = g(x) $$

- 假设后验概率服从伯努利分布，于是：

	$$ p(y|x;\theta) = (h\_{\theta}(x))^{y} \cdot (1 - h\_{\theta}(x))^{1-y} $$

- 对数似然计算如下：

	$$
	\begin{align\*}
	log \ L(\theta) &= log \ \prod\_{i=1}^{M} \ (h\_{\theta}(x\_{i}))^{y\_{i}} \cdot (1 - h\_{\theta}(x\_{i}))^{1-y\_{i}} \newline
	&= \sum\_{i=1}^{M} log \ (h\_{\theta}(x\_{i}))^{y\_{i}} \cdot (1 - h\_{\theta}(x\_{i}))^{1-y\_{i}} \newline
	&= \sum\_{i=1}^{M} \left( y\_{i} \cdot log \ h\_{\theta}(x\_{i}) + (1 - y\_{i}) \cdot log \ (1 - h\_{\theta}(x\_{i})) \right) \newline
	\end{align\*}
	$$

- 由于最大化对数似然函数等于最小化损失函数，损失函数计算如下：

	$$ J(\theta) = - \sum\_{i=1}^{M} \left( y\_{i} \cdot log \ h\_{\theta}(x\_{i}) + (1 - y\_{i}) \cdot log \ (1 - h\_{\theta}(x\_{i})) \right) $$

- 令 \\(X = \\left\[ \begin{matrix} x\_{10} & x\_{11} & \cdots & x\_{1N} \\\\ x\_{20} & x\_{21} & \cdots & x\_{2N} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ x\_{M0} & x\_{M1} & \cdots & x\_{MN} \end{matrix} \\right\], \ \theta = \\left\[ \begin{matrix} \theta\_{0} \\\\ \theta\_{1} \\\\ \vdots \\\\ \theta\_{N} \end{matrix} \\right\], \ Y = \\left\[ \begin{matrix} y\_{1} \\\\ y\_{2} \\\\ \vdots \\\\ y\_{M} \end{matrix} \\right\] \\)，则 \\(\frac{1}{1 + e^{-X\theta}}-Y = \\left\[ \begin{matrix} \frac{1}{1 + e^{-x\_{1}^{T}\theta}}-y\_{1} \\\\ \frac{1}{1 + e^{-x\_{2}^{T}\theta}}-y\_{2} \\\\ \vdots \\\\ \frac{1}{1 + e^{-x\_{M}^{T}\theta}}-y\_{M} \end{matrix} \\right\]\\)

### 优化求解

#### 梯度下降

- 通过梯度下降法求解 \\(\theta\_{j} \leftarrow \eta \cdot \nabla\_{\theta\_{j}}J(\theta)\\)：

	$$
	\begin{align\*}
	\nabla\_{\theta\_{j}}J(\theta) &= -\sum\_{i=1}^{M} \left( y\_{i} \cdot \frac{1}{h\_{\theta}(x\_{i})} \cdot \frac{\partial{h\_{\theta}(x\_{i})}}{\partial{\theta\_{j}}} + (1 - y\_{i}) \cdot \frac{1}{1 - h\_{\theta}(x\_{i})} \cdot \frac{\partial{(1 - h\_{\theta}(x\_{i}))}}{\partial{\theta\_{j}}} \right) \newline
	&= -\sum\_{i=1}^{M} \left( y\_{i} \cdot \frac{1}{h\_{\theta}(x\_{i})} + (y\_{i} - 1) \cdot \frac{1}{1 - h\_{\theta}(x\_{i})} \right) \cdot \frac{\partial{h\_{\theta}(x\_{i})}}{\partial{\theta\_{j}}} \newline
	&= -\sum\_{i=1}^{M} \left( y\_{i} \cdot \frac{1}{h\_{\theta}(x\_{i})} + (y\_{i} - 1) \cdot \frac{1}{1 - h\_{\theta}(x\_{i})} \right) \cdot \frac{\partial{h\_{\theta}(x\_{i})}}{\partial{g(x\_{i})}} \cdot \frac{\partial{g(x\_{i})}}{\partial{\theta\_{j}}} \newline
	&= -\sum\_{i=1}^{M} \left( y\_{i} \cdot \frac{1}{h\_{\theta}(x\_{i})} + (y\_{i} - 1) \cdot \frac{1}{1 - h\_{\theta}(x\_{i})} \right) \cdot h\_{\theta}(x\_{i}) \cdot (1 - h\_{\theta}(x\_{i})) \cdot x\_{ij} \newline
	&= \sum\_{i=1}^{M} (h\_{\theta}(x\_{i}) - y\_{i}) \cdot x\_{ij} \newline
	&= \sum\_{i=1}^{M} (\frac{1}{1 + e^{-\theta^{T}x\_{i}}} - y\_{i}) \cdot x\_{ij} \newline
	\end{align\*}
	$$

- 向量化如下：

	$$ \theta \leftarrow \theta - \eta \cdot X^{T} \left( \frac{1}{1 + e^{-X\theta}} - Y \right) $$

- 与最小二乘相比，梯度的形式相同，\\(h\_{\theta}(x)\\) 的计算方式不同

### 对数损失

- 对数损失函数源于最大似然估计

- 如果非要使用平方损失函数，会导致损失函数非凸。只考虑单个样本，证明如下：

	- 对于对数损失，一阶导数如下：

		$$ \frac{\partial{J(\theta)}}{\partial{\theta\_{j}}} = (h\_{\theta}(x) - y) \cdot x\_{j} $$

		- 进一步，二阶导数如下：

			$$ \frac{\partial^{2}(J(\theta))}{\partial{\theta\_{j}^{2}}} = h\_{\theta}(x) \cdot (1 - h\_{\theta}(x)) \cdot x\_{j}^{2} $$

		- 由于 sigmoid 导数大于 \\(0\\)，二阶导数恒为正；因此对数损失为凸函数，可以收敛到全局最小值

	- 对于平方损失，一阶导数如下：

		$$ \frac{\partial{J(\theta)}}{\partial{\theta\_{j}}} = (h\_{\theta}(x) - y) \cdot h\_{\theta}(x) \cdot (1 - h\_{\theta}(x)) \cdot x\_{j} $$

		- 进一步，二阶导数如下：

			$$ \frac{\partial^{2}(J(\theta))}{\partial{\theta\_{j}^{2}}} = \left( 2h(y+1) - 3h^{2} - y \right) \cdot h \cdot (1 - h) \cdot x\_{j}^{2} $$
			
			- 尽管 sigmoid 导数大于 \\(0\\)，但不能保证 \\(2h(y+1) - 3h^{2} - y \geq 0\\)；因此平方损失非凸，不保证收敛到全局最小值