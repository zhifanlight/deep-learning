<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 高数基础

## Jensen 不等式

- 向下凸起的函数称为凸函数，凸函数有最小值

- 对于凸函数上的任意两点，这两点的割线一定在这两点间函数图像的上方，即：

	$$ a f(x\_{1}) + b f(x\_{2}) \geq f(a x\_{1} + b x\_{2}) $$
	
	- 其中 \\(0 \leq a,b \leq 1\\) 且 \\(a + b = 1\\)

	![img](images/convex.png)

## 泰勒展开

- 函数 \\(f(x)\\) 的在 \\(x=a\\) 处的泰勒展开如下：

	$$ f(x) = \sum\_{n=0}^{\infty} \frac{f^{n}(a)}{n!} (x-a)^{n} $$

## 三次方因式分解

$$ a^{3} - b^{3} = (a-b) \cdot \left( a^{2} + b^{2} + ab \right) $$

$$ a^{3} + b^{3} = (a+b) \cdot \left( a^{2} + b^{2} + ab \right) $$
 
## 分步积分

$$ \int f(x) \cdot g'(x) \ dx = f(x) \cdot g(x) - \int f'(x) \cdot g(x) \ dx $$

## 极值点 VS 驻点 VS 鞍点 VS 拐点

### 极值点

- 对于 \\(x\_{0}\\) 的去心邻域 \\((x\_{0}-\epsilon, \ x\_{0}+\epsilon)\\)：

	- 如果 \\(f(x\_{0}) > f(x)\\) 恒成立，则 \\((x\_{0}, \ f(x\_{0})\\) 是极大值点

	- 如果 \\(f(x\_{0}) < f(x)\\) 恒成立，则 \\((x\_{0}, \ f(x\_{0})\\) 是极小值点

- 极值点处的导数可能不存在

### 驻点

- 一阶导数存在且为 \\(0\\) 的点称为驻点

### 鞍点

- 在一维空间中，鞍点即驻点

- 在高维空间中，鞍点是指在一个方向是极小值，在另一个方向是极大值的点

### 拐点

- 二阶导数为 \\(0\\) 且两侧二阶导数异号的点称为拐点

## 偏导数 VS 梯度 VS 方向导数

### 偏导数

- 多元函数在 \\(x\\) 处关于某一个特定变量 \\(x\_{i}\\) 的导数称为偏导数

### 梯度

- 梯度 \\(\nabla\_{x} f(x)\\) 的第 \\(i\\) 个元素是 \\(f\\) 对 \\(x\_{i}\\) 的偏导数

- 沿着梯度方向，函数上升最快；沿着负梯度方向，函数下降最快

### 方向导数

- 函数 \\(f\\) 在单位向量 \\(u\\) 方向（任意方向）上的斜率

- 函数 \\(f(x+\beta u)\\) 关于 \\(\beta\\)（\\(\beta \rightarrow 0\\)）的导数：

	$$ \lim\_{\beta \rightarrow 0} \frac{f(x+\beta u) - f(x)}{\beta} $$

	- 退化为一维时计算如下：

		$$ \lim\_{\Delta x \rightarrow 0} \frac{f(x+\Delta x * 1) - f(x)}{\Delta x} $$

- 方向导数计算如下：

	$$ \lim\_{\beta \rightarrow 0} \frac{\partial f(x+\beta u)}{\partial \beta} = u^{T} \nabla\_{x} f(x) $$

- 寻找使 \\(f\\) 下降最快的方向：

	$$ \min\_{u, ||u||=1} u^{T} \nabla\_{x} f(x) = \min\_{u, ||u||=1} ||u|| \cdot ||\nabla\_{x} f(x)|| \cdot \cos \theta $$
	
	- 其中，\\(\theta\\) 是单位向量 \\(u\\) 与梯度 \\(\nabla\_{x} f(x)\\) 的夹角

- 代入约束条件 \\(||u||=1\\)，并忽略与 \\(u\\) 无关的梯度项：

	$$ \min\_{u} \cos \theta $$

	- 当 \\(\theta=180^{\circ}\\) 时取得最小值；即沿着负梯度方向下降最快

## 常用函数求导

- 对数：

	$$ \frac{d \ \log\_{a}x}{d \ x} = \frac{1}{x \ \ln \ a} $$

## 链式法则

- 假设 \\(z = f(x, y)\\)，其中 \\(x = g(t), \ y = h(t)\\)，那么：

	$$ \frac{\partial{z}}{\partial{t}} = \frac{\partial{z}}{\partial{x}} \cdot \frac{\partial{x}}{\partial{t}} + \frac{\partial{z}}{\partial{y}} \cdot \frac{\partial{y}}{\partial{t}}$$