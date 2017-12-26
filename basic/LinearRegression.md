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

	$$ J(\theta) = \frac{1}{M} \sum\_{i=1}^{M} \left[h(x\_{i}) - y\_{i}\right]^{2} = \frac{1}{M} \sum\_{i=1}^{M} \left[ \theta^{T}x\_{i} - y\_{i} \right]^{2} $$

### 优化求解

#### 梯度下降

- 参见 [Optimizer.md](Optimizer.md)

- 通过梯度方法求解每一个 \\(\theta\_{i}\\)

#### 正规方程

- 令 \\(X = \\left\[ \begin{matrix} x\_{10}, x\_{11}, ..., x\_{1N} \\\\ x\_{20}, x\_{21}, ..., x\_{2N} \\\\ ... \\\\ x\_{M0}, x\_{M1}, ..., x\_{MN} \end{matrix} \\right\], \ Y = \\left\[ \begin{matrix} y\_{1} \\\\ y\_{2} \\\\ ... \\\\ y\_{M} \end{matrix} \\right\] \\)，则 \\(X\theta-Y = \\left\[ \begin{matrix} x\_{1}^{T}\theta - y\_{1} \\\\ x\_{2}^{T}\theta-y\_{2} \\\\ ... \\\\ x\_{M}^{T}\theta-y\_{M} \end{matrix} \\right\]\\)

- 由于 \\(X\theta-Y\\) 为 \\(M + 1\\) 维列向量：

	$$ \min\_{\theta} J(\theta) = \min\_{\theta} \frac{1}{2} (X\theta-Y)^{T}(X\theta-Y) $$

- 对 \\(J(\theta)\\) 求导可得：

	$$ \nabla\_{\theta}J(\theta) = X^{T}X\theta - X^{T}Y $$

- 因此，最优解：

	$$ \arg\min\_{\theta}J(\theta) = \left\(X^{T}X\right\)^{-1}X^{T}Y $$