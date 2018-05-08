<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 核函数

## 背景介绍

- 在原始空间线性不可分的问题在高维空间可分，从低维空间变换到高维空间需要映射

- 核函数可以用于 SVM 中，但不仅限于用在 SVM 中

## 数学推导

- 假设 \\(\phi(x)\\) 表示原始空间向高维空间的映射

- 在使用核函数时，通常具有 \\(x\_{i}^{T} x\_{j}\\) 的形式，用 \\(\phi(x\_{i}), \ \phi(x\_{j})\\) 分别替代 \\(x\_{i}, \ x\_{j}\\) 即可实现向高维空间的映射

- 由于 \\(\phi(x)\\) 是高维甚至无穷维，计算不变，可以通过引入核函数 \\(K\\) 直接计算乘积 \\(\phi(x\_{i})^{T} \phi(x\_{j})\\)：

	$$ K(x\_{i}, x\_{j}) = \phi(x\_{i})^{T} \phi(x\_{j}) $$

## 常用核函数

### 多项式核

$$ K(x\_{i},x\_{j}) = \left( x\_{i}^{T} x\_{j} \right)^{d} \qquad d \geq 1 $$

- 多项式核将数据从原始空间映射到更高维：

	- 以 \\(K(x,y) = \left( x^{T}y \right)^{2} \\) 对二维向量 \\((x,y)\\) 进行映射为例，展开可得：

	$$ K(x,y) = \left( x\_{1} x\_{2} + y\_{1} y\_{2} \right)^{2} = x\_{1}^{2} x\_{2}^{2} + 2x\_{1}x\_{2}y\_{1}y\_{2} + y\_{1}^{2}y\_{2}^{2} $$
	
	- 此时特征映射函数为：

		$$ \Phi(x,y) = (x^{2}, \ \sqrt{2}xy, \ y^{2}) $$
		
		- 实现了二维空间到三维空间的映射

### 线性核

$$ K(x\_{i},x\_{j}) = x\_{i}^{T} x\_{j} $$

### 高斯核（RBF 核）

$$ K(x\_{i},x\_{j}) = \exp \left( -\frac{||x\_{i} - x\_{j}||^{2}}{2\sigma^{2}} \right) \qquad \sigma > 0 $$

- 高斯核将数据从原始空间映射到无穷维：

	- 高斯核带宽 \\(\sigma\\) 不影响数据维度，对高斯核化简可得：

		$$ K(x,y) = \exp \left( -||x||^{2} \right) \cdot \exp \left( -||y||^{2} \right) \cdot \exp \left( 2x^{T}y \right) $$
		
	- 根据泰勒公式：

		$$ K(x,y) = \exp \left( -||x||^{2} \right) \cdot \exp \left( -||y||^{2} \right) \cdot \sum\_{n=0} \frac{\left( 2x^{T}y \right)^{n}}{n!} $$
		
		- 高斯核本质是无穷多个多项式核累加

		- 由多项式核将数据映射到更高维可知，高斯核可以将数据映射到无穷维

### 拉普拉斯核

$$ K(x\_{i},x\_{j}) = \exp \left( -\frac{||x\_{i} - x\_{j}||}{\sigma} \right) \qquad \sigma > 0 $$

### Sigmoid 核

$$ K(x\_{i},x\_{j}) = tanh \left( \beta x\_{i}^{T} x\_{j} + \theta \right) \qquad \beta > 0, \ \theta < 0 $$