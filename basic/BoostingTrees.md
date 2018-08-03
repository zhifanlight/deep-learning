<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 提升树

## 背景介绍

- 以分类树或回归树为基本分类器的提升方法

- 已知训练集 \\(\\{(x\_{1},y\_{1}), \ (x\_{2},y\_{2}), \ \cdots, \ (x\_{M},y\_{M})\\}\\)

## 提升树

### 二叉分类树

- 二分类问题使用指数损失函数，只需将 Adaboost 中的分类器限制为二叉决策树即可

- 参考 [Adaboost.md](Adaboost.md)

### 二叉回归树

- 回归问题使用平方损失函数，使用 CART 回归树

- 如果将输入空间划分为 \\(J\\) 个不相交区域 \\(R\_{1}, \ R\_{2}, \ \cdots, \ R\_{J}\\)，且在每个区域上的输出为常量 \\(c\_{j}\\)，由此得到的提升树可表示为

	$$ T(x;\theta) = \sum\_{j=1}^{J} c\_{j} \cdot I(x \in R\_{j}) $$

#### 数学推导

- 在算法的第 \\(t\\) 步，给定当前模型 \\(f\_{t-1}(x)\\)，需要求解：

	$$ \theta\_{t}^{\*} = \arg \min\_{\theta} \sum\_{i=1}^{M} L \left( y\_{i}, \ f\_{t-1}(x\_{i}) + T(x\_{i};\theta\_{t}) \right) $$

- 对于平方损失函数：

	$$ L(y\_{i}, \ f\_{t-1}(x\_{i}) + T(x\_{i};\theta)) = \left[ y\_{i} - f\_{t-1}(x\_{i}) - T(x\_{i}; \theta) \right]^{2} = \left[ r\_{t, \ i} - T(x\_{i}; \theta) \right]^{2} $$
	
	- 其中 \\(r\_{t, \ i}\\) 表示本轮（第 \\(t\\) 轮）开始时 \\(x\_{i}\\) 的残差

	- 因此当前模型是在拟合上一轮的残差

#### 算法流程

- 初始化：

	$$ f\_{0}(x) = 0 $$

- 对于第 \\(1, \ 2, \cdots, \ T\\) 轮：

	- 计算每一个样本的残差：

		$$ r\_{t, \ i} = y\_{i} - f\_{t-1}(x\_{i}) $$

	- 拟合残差 \\(r\_{t, \ i}\\) 得到一棵回归树：

		$$ T(x; \theta\_{t}) = \sum\_{j=1}^{J} c\_{j} \cdot I(x \in R\_{j}) $$

	- 更新回归树：

		$$ f\_{t}(x) = f\_{t-1}(x) + T(x; \theta\_{t}) $$

- 得到最终的提升树：

	$$ f\_{T}(x) = \sum\_{t=1}^{T} T(x; \theta\_{t}) $$

## GBDT

- Gradient Boosting Decision Tree，即梯度提升决策树

- 当使用指数损失函数（分类）和平方损失函数（回归）时，提升树的优化都比较简单；但对于一般的损失函数，不太容易优化

- GBDT 利用损失函数在当前模型的负梯度（响应值）来代替残差：

	$$ r\_{t, \ i} = - \left[ \frac{\partial{L \left( y\_{i}, \ f\_{t-1}(x\_{i}) \right)}}{\partial{f\_{t-1}(x\_{i})}} \right] $$

	- 当使用平方损失函数时，\\(r\_{t, \ i}\\) 即为实际的残差

### 算法流程

- 初始化：

	$$ f\_{0}(x) = \arg \min\_{c} \sum\_{i=1}^{M} L(y\_{i}, \ c) $$

- 对于第 \\(1, \ 2, \cdots, \ T\\) 轮：

	- 计算每一个样本的响应值：

		$$ r\_{t, \ i} = - \left[ \frac{\partial{L \left( y\_{i}, \ f\_{t-1}(x\_{i}) \right)}}{\partial{f\_{t-1}(x\_{i})}} \right] $$

	- 拟合响应值 \\(r\_{t, \ i}\\) 得到一棵回归树：

		$$ T(x; \theta\_{t}) = \sum\_{j=1}^{J} c\_{j} \cdot I(x \in R\_{j}) $$
	
	- 更新回归树：

		$$ f\_{t}(x) = f\_{t-1}(x) + T(x; \theta\_{t}) $$

- 得到最终的 GBDT：

	$$ f\_{T}(x) = \sum\_{t=1}^{T} T(x; \theta\_{t}) $$

## XGBoost

- 相对 GBDT，基分类器除了回归树，还支持线性分类器

	- 分类问题：带 \\(L\_{1}\\) 和 \\(L\_{2}\\) 正则项的逻辑回归

	- 回归问题：带 \\(L\_{1}\\) 和 \\(L\_{2}\\) 正则项的线性回归

- 显式将树的复杂度作为正则项加入优化目标：

	$$ \Omega(f\_{t}) = \frac{\gamma}{2} \ J + \frac{\lambda}{2} \sum\_{j=1}^{J} b\_{j}^{2} $$

	- 其中 \\(J\\) 为回归树中叶结点的数量，\\(b\_{j}\\) 表示第 \\(j\\) 个叶结点的输出

- 权重衰减：在进行完一次迭代后，将叶结点的权值乘上衰减系数，以削弱每棵树的影响，让后面有更大的学习空间

- 支持自定义损失函数，优化过程同时用到一阶导数和二阶导数：

	$$ L\_{t} \approx \sum\_{i=1}^{M} \left( L \left( y\_{i}, f\_{t-1}(x\_{i}) \right) + g\_{i} \cdot T(x; \theta) + \frac{1}{2} h\_{i} \cdot T^{2}(x; \theta) \right) + \Omega(f\_{t}) $$
	
	- 其中，\\(g\_{i}, \ h\_{i}\\) 分别为损失函数对当前模型的一阶导数、二阶导数：

		$$ g\_{i} = \frac{\partial{L(y\_{i}, \ f\_{t-1}(x))}}{\partial{f\_{t-1}}} \qquad g\_{i} = \frac{\partial^{2}{L(y\_{i}, \ f\_{t-1}(x))}}{\partial^{2}{f\_{t-1}}} $$