<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Softmax 回归

## 背景介绍

- softmax 回归用于多分类任务，其结果是多个超平面

- 对于多分类任务，假设获得了 \\(N\\) 维空间中的 \\(M\\) 个点及观察值：\\((x\_{1},y\_{1}), (x\_{2},y\_{2}), \cdots, (x\_{M},y\_{M})\\)，其中 \\(x\_{i}\\) 是 \\(N\\) 维列向量 \\(\\left\[ \begin{matrix} x\_{i1} \\\\ x\_{i2} \\\\ \vdots \\\\ x\_{iN} \end{matrix} \\right\]\\)，\\(y\_{i}\\) 是标量

## 数学推导

### 模型建立

- 如果把 \\(N\\) 维列向量 \\(x\_{i}\\) 扩充到 \\(N + 1\\) 维并令 \\(x\_{i0}=1\\)，那么第 \\(i\\) 个待拟合超平面可以定义为 \\(\theta\_{i}^{T}x\\)，其中 \\(\theta\_{i}\\) 是 \\(N + 1\\) 维列向量

- 假设后验概率服从多项式分布，于是：

	$$ p(y|x;\theta) = \prod\_{i=1}^{K} \phi\_{i}^{I(y=i)} $$
	
	- \\(\phi\_{i}\\) 表示 \\(y=i|x\\) 的概率：
	
		$$ \phi\_{i} = \frac{e^{\theta\_{i}^{T}x}}{\sum\_{j=1}^{K}e^{\theta\_{j}^{T}x}} $$

	- \\(I(c)\\) 是指示函数：
	
		$$ I(c)=\\left\\{ \begin{matrix} 1, \quad if \ c \ is \ true \\\\ 0, \quad if \ c \ is \ false \end{matrix} \\right. $$

- 对数似然计算如下：

	$$
	\begin{align\*}
	log \ L(\theta) &= log \ \prod\_{i=1}^{M} \prod\_{j=1}^{K} \ \left( \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \right)^{I(y\_{i}=j)} \newline
	&= \sum\_{i=1}^{M} \sum\_{j=1}^{K} I(y\_{i}=j) \cdot log \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \newline
	\end{align\*}
	$$

- 由于最大化对数似然函数等于最小化损失函数，损失函数计算如下：

	$$ J(\theta) = - \sum\_{i=1}^{M} \sum\_{j=1}^{K} I(y\_{i}=j) \cdot log \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} $$

- 令 \\(X = \\left\[ \begin{matrix} x\_{10} & x\_{11} & \cdots & x\_{1N} \\\\ x\_{20} & x\_{21} & \cdots & x\_{2N} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ x\_{M0} & x\_{M1} & \cdots & x\_{MN} \end{matrix} \\right\], \ \theta = \\left\[ \begin{matrix} \theta\_{10} & \theta\_{20} & \cdots & \theta\_{K0} \\\\ \theta\_{11} & \theta\_{21} & \cdots & \theta\_{K1} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ \theta\_{1N} & \theta\_{2N} & \cdots & \theta\_{KN} \end{matrix} \\right\], \ Y = \\left\[ \begin{matrix} y\_{1} \\\\ y\_{2} \\\\ \vdots \\\\ y\_{M} \end{matrix} \\right\], \ e^{X\theta} = \\left\[ \begin{matrix} e^{\theta\_{1}^{T}x\_{1}} & e^{\theta\_{2}^{T}x\_{1}} & \cdots & e^{\theta\_{K}^{T}x\_{1}} \\\\ e^{\theta\_{1}^{T}x\_{2}} & e^{\theta\_{2}^{T}x\_{2}} & \cdots & e^{\theta\_{K}^{T}x\_{2}} \\\\ \vdots & \vdots & \ddots & \vdots \\\\ e^{\theta\_{1}^{T}x\_{M}} & e^{\theta\_{2}^{T}x\_{M}} & \cdots & e^{\theta\_{K}^{T}x\_{M}} \end{matrix} \\right\] \\)

- \\(I\\) 表示 \\(Y\\) 的 \\(OneHot\\) 结果，\\(norm(e^{X\theta})\\) 表示 \\(e^{X\theta}\\) 按行归一化的结果

### 优化求解

#### 梯度下降

- 通过梯度下降法求解 \\(\theta\_{jk} \leftarrow \eta \cdot \nabla\_{\theta\_{jk}}J(\theta)\\)：

	$$
	\begin{align\*}
	\nabla\_{\theta\_{jk}}J(\theta) &= - \left( \sum\_{i=1}^{M} I(y\_{i}=j) \cdot \frac{\partial \theta\_{j}^{T}x\_{i}}{\partial \theta\_{jk}} - \sum\_{i=1}^{M} I(y\_{i}=j) \cdot \frac{\partial log \sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}}{\partial \theta\_{jk}} - \sum\_{i=1}^{M} \sum\_{c \neq j}^{K} I(y\_{i}=c) \cdot \frac{\partial log \sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}}{\partial \theta\_{jk}} \right) \newline
	&= - \left( \sum\_{i=1}^{M} I(y\_{i}=j) \cdot x\_{ik} - \sum\_{i=1}^{M} I(y\_{i}=j) \cdot \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \cdot x\_{ik} - \sum\_{i=1}^{M} \sum\_{c \neq j}^{K} I(y\_{i}=c) \cdot \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \cdot x\_{ik} \right) \newline
	&= - \left( \sum\_{i=1}^{M} I(y\_{i}=j) \cdot x\_{ik} - \sum\_{i=1}^{M} I(y\_{i}=j) \cdot \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \cdot x\_{ik} - \sum\_{i=1}^{M} \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \cdot x\_{ik} \cdot \sum\_{c \neq j}^{K} I(y\_{i}=c) \right) \newline
	&= - \left( \sum\_{i=1}^{M} I(y\_{i}=j) \cdot x\_{ik} - \sum\_{i=1}^{M} I(y\_{i}=j) \cdot \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \cdot x\_{ik} - \sum\_{i=1}^{M} \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \cdot x\_{ik} \cdot (1 - I(y\_{i}=j)) \right) \newline
	&= - \left( \sum\_{i=1}^{M} I(y\_{i}=j) \cdot x\_{ik} - \sum\_{i=1}^{M} \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \cdot x\_{ik} \right) \newline
	&= - \sum\_{i=1}^{M} \left( x\_{ik} \cdot \left( I(y\_{i}=j) - \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \right) \right) \newline
	\end{align\*}
	$$

- 向量化如下：

	$$ \theta\_{j} \leftarrow \theta\_{j} + \eta \cdot \sum\_{i=1}^{M} \left( x\_{i} \cdot \left( I(y\_{i}=j) - \frac{e^{\theta\_{j}^{T}x\_{i}}}{\sum\_{s=1}^{K} e^{\theta\_{s}^{T}x\_{i}}} \right) \right) $$

- 矩阵化如下：

	$$ \theta \leftarrow \theta + \eta \cdot X^{T} \left( I - norm(e^{X\theta}) \right) $$