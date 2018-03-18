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

### 线性核

$$ K(x\_{i},x\_{j}) = x\_{i}^{T} x\_{j} $$

### 高斯核（RBF 核）

$$ K(x\_{i},x\_{j}) = exp \left( -\frac{||x\_{i} - x\_{j}||^{2}}{2\sigma^{2}} \right) \qquad \sigma > 0 $$

### 拉普拉斯核

$$ K(x\_{i},x\_{j}) = exp \left( -\frac{||x\_{i} - x\_{j}||}{\sigma} \right) \qquad \sigma > 0 $$

### Sigmoid 核

$$ K(x\_{i},x\_{j}) = tanh \left( \beta x\_{i}^{T} x\_{j} + \theta \right) \qquad \beta > 0, \ \theta < 0 $$