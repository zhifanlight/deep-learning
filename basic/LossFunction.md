<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 损失函数

## 背景介绍

- 非负实值函数，用来估计模型的预测值 \\(f(x)\\) 对真实值 \\(y\\) 的拟合程度；损失函数越小，模型越好；通常用 \\(L(y, \ f(x))\\) 表示。

- 一般由经验风险项、正则化项组成，定义如下：

	$$ J(\theta) = \frac{1}{N} \sum\_{i=1}^{N} L(y\_{i}, \ f(x\_{i}; \theta)) + \lambda \cdot R(\theta) $$

- 关于正则化，参考 [Regularization.md](Regularization.md)

## 常用损失函数

### 0-1 损失函数

- 对于分类标签 \\(f(x\_{i})\\) 与真实标签 \\(y\_{i}\\)，相同时为 \\(0\\)，不同时为 \\(1\\)：

	$$ L(y\_{i}, f(x\_{i})) = \left\\{ \begin{matrix} 1, \quad f(x\_{i}) \neq y\_{i} \\\\ 0, \quad f(x\_{i}) = y\_{i} \end{matrix} \\right. $$

- 不易求导，因此在模型训练时不常用；通常用于模型的评价

### 平方损失函数

- 将预测值 \\(f(x\_{i})\\) 与真实值 \\(y\_{i}\\) 之间的欧式距离作为误差：

	$$ L(y\_{i}, f(x\_{i})) = (y\_{i} - f(x\_{i}))^{2} $$

	- 用于在线性回归，参考 [LinearRegression.md](LinearRegression.md)

### 对数损失函数

- 来源于最大似然估计：最大化对数似然等于最小化损失函数

- 将预测类别 \\(f(x\_{i})\\) 与真实类别 \\(y\_{i}\\) 之间的交叉熵作为误差：

	- 二分类：

		$$ p(i) = sigmoid \ (f(x\_{i})) $$

		$$ L(y\_{i}, f(x\_{i})) = -( y\_{i} \cdot log \ p\_{i} + (1 - y\_{i}) \cdot log \ (1 - p\_{i}) ) $$
		
		- 用于逻辑回归，参考 [LogisticRegression.md](LogisticRegression.md)

	- 多分类：

		$$ p(i) = softmax \ (f(x\_{i})) $$

		$$ L(y\_{i}, f(x\_{i})) = - \sum\_{j} y\_{ij} \cdot log \ p\_{ij} $$
		
		- 用于 Softmax 回归，参考 [SoftmaxRegression.md](SoftmaxRegression.md)

### 指数损失函数

- 标准形式如下：

	$$ L(y\_{i}, f(x\_{i})) = e^{-y\_{i} \cdot f(x\_{i})} $$

	- 用于 Adaboost，参考 [Adaboost.md](Adaboost.md)

### Hinge 损失函数

- 标准形式如下：

	$$ L(y\_{i}, \ f(x\_{i})) = max(0, 1 - y\_{i} \cdot f(x\_{i})) $$
	
	- 用于软间隔 SVM，参考 [SVM.md](SVM.md)

		- 由 \\(\xi\_{i}\\) 约束条件可得：

			$$ \xi\_{i} \geq max(0, 1 - y\_{i} \cdot h(x\_{i})) $$
		
		- 代入目标函数可得：

			$$ \min\_{w,b,\xi} \frac{1}{2} ||w||^{2} + C \sum\_{i=1}^{M} max(0, 1 - y\_{i} \cdot h(x\_{i})) $$
		
		- 因此软间隔 SVM 的损失函数为 Hinge 损失与 \\(L\_{2}\\) 正则项之和