<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 线性回归

## 背景介绍

- 在回归任务中，最基础的模型就是线性回归，即输入值和输出值之间存在简单的线性关系：

	- 在二维平面中，线性是一条直线

	- 在三维空间中，线性是一个平面

	- 在多维空间中，线性是一个超平面

- 对于线性回归任务，假设获得了 \\(N\\) 维空间中的 \\(M\\) 个点及观察值：\\((x\_{1},y\_{1}), (x\_{2},y\_{2}), ..., (x\_{M},y\_{M})\\)，其中 \\(x\_{i}\\) 是 \\(N\\) 维列向量 \\(\\left\[ \begin{matrix} x\_{i1} \\\\ x\_{i2} \\\\ ... \\\\ x\_{iN} \end{matrix} \\right\]\\)，\\(y\_{i}\\) 是标量

- 对于这些点，最好的拟合超平面要求总的误差最小，有以下三个标准可以选择：

	- 误差和最小：容易使得正、负误差互相抵消
	
	- 误差绝对值和最小：难以优化求解

	- 误差平方和最小：计算方便，反映欧式距离

## 数学推导

### 模型建立

- 采用误差平方和时，通常使用最小二乘法进行拟合

- 如果把 \\(N\\) 维列向量 \\(x\_{i}\\) 扩充到 \\(N + 1\\) 维并令 \\(x\_{i0}=1\\)，那么待拟合超平面可以定义为 \\(h(x) = \theta^{T}x\\)，其中 \\(\theta\\) 是 \\(N + 1\\) 维列向量，则损失函数定义如下：

	$$ J(\theta) = \frac{1}{2} \sum\_{i=1}^{M} \left( h(x\_{i}) - y\_{i} \right)^{2} = \frac{1}{2} \sum\_{i=1}^{M} \left( \theta^{T}x\_{i} - y\_{i} \right)^{2} $$

- 令 \\(X = \\left\[ \begin{matrix} x\_{10}, x\_{11}, ..., x\_{1N} \\\\ x\_{20}, x\_{21}, ..., x\_{2N} \\\\ ... \\\\ x\_{M0}, x\_{M1}, ..., x\_{MN} \end{matrix} \\right\], \ \theta = \\left\[ \begin{matrix} \theta\_{0} \\\\ \theta\_{1} \\\\ ... \\\\ \theta\_{N} \end{matrix} \\right\], \ Y = \\left\[ \begin{matrix} y\_{1} \\\\ y\_{2} \\\\ ... \\\\ y\_{M} \end{matrix} \\right\] \\)，则 \\(X\theta-Y = \\left\[ \begin{matrix} x\_{1}^{T}\theta - y\_{1} \\\\ x\_{2}^{T}\theta-y\_{2} \\\\ ... \\\\ x\_{M}^{T}\theta-y\_{M} \end{matrix} \\right\]\\)

### 优化求解

#### 梯度下降

- 通过梯度下降法求解 \\(\theta\_{j} \leftarrow \eta \cdot \nabla\_{\theta\_{j}}J(\theta)\\)：

	$$
	\begin{align\*}
	\nabla\_{\theta\_{j}}J(\theta) &= \sum\_{i=1}^{M} \left( \theta^{T}x\_{i} - y\_{i} \right) \cdot \frac{\partial{\left( \theta^{T}x\_{i} - y\_{i} \right)}}{\partial{\theta\_{j}}} \newline
	&= \sum\_{i=1}^{M} \left( \theta^{T}x\_{i} - y\_{i} \right) \cdot x\_{ij} \newline
	&= \sum\_{i=1}^{M} \left( h(x\_{i}) - y\_{i} \right) \cdot x\_{ij} \newline
	\end{align\*}
	$$

- 向量化如下：

	$$ \theta \leftarrow \theta - \eta \cdot X^{T} ( X\theta - Y ) $$

#### 正规方程

- 由于 \\(X\theta-Y\\) 为 \\(M\\) 维列向量：

	$$ \min\_{\theta} J(\theta) = \min\_{\theta} \frac{1}{2} (X\theta-Y)^{T}(X\theta-Y) $$

- 对 \\(J(\theta)\\) 求导可得：

	$$ \nabla\_{\theta}J(\theta) = X^{T}X\theta - X^{T}Y $$

- 令导数为 0 可得最优解：

	$$ \theta^{\*} = \left\(X^{T}X\right\)^{-1}X^{T}Y $$

## 最小二乘概率解释

### 允许误差存在

- 由于所有预测值都不可能完美地与真实值契合，误差必然存在；而拟合的目标就是让误差尽可能小

- 因此，输入值 \\(x\_{i}\\) 和输出值 \\(y\_{i}\\) 之间的关系可表示为：

	$$ y\_{i} = \theta^{T} x\_{i} + \epsilon\_{i} $$

### 假设误差分布

- 误差项 \\(\epsilon\_{i}\\) 捕捉未被设置为特征的变量，假设其独立同分布，且服从高斯分布 \\(N(0, \sigma^{2})\\)：

	$$ p(\epsilon\_{i}) = \frac{1}{\sqrt{2 \pi} \sigma} exp \left(- \frac{\epsilon\_{i}^{2}}{2 \sigma^{2}} \right) $$

- 在以 \\(\theta\\) 为参数且 \\(x\_{i}\\) 给定时，\\(\theta^{T} x\_{i}\\) 为定值，此时的 \\(y\_{i}\\) 服从高斯分布 \\(N(\theta^{T} x\_{i}, \sigma^{2})\\)：

	$$ p(y\_{i}|x\_{i};\theta) = \frac{1}{\sqrt{2 \pi} \sigma} exp \left(- \frac{ \left( y\_{i} - \theta^{T} x\_{i} \right)^{2}}{2 \sigma^{2}} \right) $$

### 最大似然估计

- 对数似然计算如下：

	$$
	\begin{align\*}
	log \ L(\theta) &= log \ \prod\_{i=1}^{m} \frac{1}{\sqrt{2 \pi} \sigma} exp \left(- \frac{ \left( y\_{i} - \theta^{T} x\_{i} \right)^{2}}{2 \sigma^{2}} \right) \newline
	&= \sum\_{i=1}^{m} log \ \frac{1}{\sqrt{2 \pi} \sigma} exp \left(- \frac{ \left( y\_{i} - \theta^{T} x\_{i} \right)^{2}}{2 \sigma^{2}} \right) \newline
	&= m \ log \ \frac{1}{\sqrt{2 \pi} \sigma} - \frac{1}{\sigma^{2}} \cdot \frac{1}{2} \sum\_{i=1}^{m} \left( y\_{i} - \theta^{T} x\_{i} \right)^{2} \newline
	\end{align\*}
	$$

- 最小化损失函数，实质上是最大化对数似然函数：

	$$ min \ \ J(\theta) = max \ \ log \ L(\theta) $$