<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 支持向量机

## 背景介绍

- 对于给定数据集的分类任务，判别面通常不止一个

- SVM 通过最大化正、负支持向量的间隔，来寻找最优判别面，以提高泛化性能

## 数学推导

### 线性可分

- 假设样本集线性可分，判别面为 \\(h(x)=w^{T}x+b\\)，类别标签 \\(y\_{i} \in \\{-1,+1\\}\\)，规则如下：

	$$ \\left\\{ \begin{matrix} y\_{i}=+1 \quad \Leftrightarrow \quad w^{T}x\_{i}+b > 0 \\\\ y\_{i}=-1 \quad \Leftrightarrow \quad w^{T}x\_{i}+b < 0  \end{matrix} \\right. $$

- 统一为 \\(y\_{i} \cdot (w^{T}x\_{i}+b) > 0\\)，通过对 \\(w\\) 缩放，可进一步保证 \\(y\_{i} \cdot (w^{T}x\_{i}+b) \geq 1\\)，而满足 \\(y\_{i} \cdot (w^{T}x\_{i}+b) = 1\\) 的样本称为支持向量

- SVM 优化目标是最大化正、负支持向量间隔：

	$$ \max\_{w,b} 2 \frac{|w^{T}x\_{i} + b|}{||w||^{2}} = \max\_{w,b} \frac{2}{||w||^{2}} $$

- 由于上述目标函数求解困难，可以进行如下转换：

	$$ \max\_{w,b} \frac{2}{||w||^{2}} = \min\_{w,b} \frac{1}{2} ||w||^{2} = \min\_{w,b} \frac{1}{2} ||w||^{2} $$
	
	- 约束条件为：
		
		$$ y\_{i} \cdot (w^{T}x\_{i}+b) \geq 1, \quad i=1,2,...,M $$

- 由拉格朗日乘子法可得：

	$$ L(w,b,\alpha) = \frac{1}{2} ||w||^{2} + \sum\_{i=1}^{M} \alpha\_{i} \cdot (1 - y\_{i} \cdot (w^{T}x\_{i}+b)) $$

	- 约束条件为：

		$$ \alpha\_{i} \geq 0, \quad i=1,2,...,M $$

		$$ \alpha\_{i} \cdot (1 - y\_{i} \cdot (w^{T}x\_{i}+b)) = 0, \quad i=1,2,...,M $$

- 计算 \\(\frac{\partial L}{\partial w}, \ \frac{\partial L}{\partial b}\\) 并令偏导数为 \\(0\\) 可得：

	$$ w = \sum\_{i=1}^{M} \alpha\_{i} y\_{i} x\_{i} \qquad \sum\_{i=1}^{M} \alpha\_{i} y\_{i} = 0 $$

	- 将上述结果代入 \\(L(w,b,\alpha)\\) 可得：

		$$ L(w,b,\alpha) = \sum\_{i=1}^{M} \alpha\_{i} - \frac{1}{2} \sum\_{i=1}^{M} \sum\_{j=1}^{M} \alpha\_{i} \alpha\_{j} \cdot y\_{i} y\_{j} \cdot x\_{i}^{T} x\_{j} $$

- 根据拉格朗日乘子法对偶性，上述问题可转化为：

	$$ \max\_{\alpha} G(\alpha) = \max\_{\alpha} \left( \sum\_{i=1}^{M} \alpha\_{i} - \frac{1}{2} \sum\_{i=1}^{M} \sum\_{j=1}^{M} \alpha\_{i} \alpha\_{j} \cdot y\_{i} y\_{j} \cdot x\_{i}^{T} x\_{j} \right) $$

	- 约束条件为：

		$$ \alpha\_{i} \geq 0, \quad i=1,2,...,M $$

		$$ \sum\_{i=1}^{M} \alpha\_{i} y\_{i} = 0, \quad i=1,2,...,M $$

### 线性不可分

- 在判别面附近，如果完全按照原始约束条件，会导致 SVM 发生较大的误差

- 为让 SVM 忽略某些噪声，可以引入松弛变量 \\(\xi\_{i} \geq 0\\) 来允许错误分类发生：

	$$ y\_{i} \cdot (w^{T}x\_{i}+b) \geq 1 - \xi\_{i}, \quad i=1,2,...,M $$

	- 对 \\(\xi\_{i}\\) 添加约束如下：

		$$ \min\_{\xi} C \sum\_{i=1}^{M} \xi\_{i} $$

- 此时 SVM 的优化目标变为：

	$$ \min\_{w,b,\xi} \frac{1}{2} ||w||^{2} + C \sum\_{i=1}^{M} \xi\_{i} $$

	- \\(C\\) 是惩罚因子，\\(C\\) 越大，表明越不希望离群点出现

	- 约束条件为：

		$$ \xi\_{i} \geq 0, \quad i=1,2,...,M $$

		$$ y\_{i} \cdot (w^{T}x\_{i}+b) \geq 1 - \xi\_{i}, \quad i=1,2,...,M $$

- 由拉格朗日乘子法可得：

	$$ L(w,b,\alpha) = \frac{1}{2} ||w||^{2} + C \sum\_{i=1}^{M} \xi\_{i} + \sum\_{i=1}^{M} \alpha\_{i} \cdot (1 - \xi\_{i} - y\_{i} \cdot (w^{T}x\_{i}+b)) - \sum\_{i=1}^{M} \beta\_{i} \cdot \xi\_{i} $$

	- 约束条件为：

		$$ \alpha\_{i} \geq 0, \quad i=1,2,...,M $$
		
		$$ \beta\_{i} \geq 0, \quad i=1,2,...,M $$

		$$ \beta\_{i} \xi\_{i} = 0, \quad i=1,2,...,M $$

		$$ \alpha\_{i} \cdot (1 - \xi\_{i} - y\_{i} \cdot (w^{T}x\_{i}+b)) = 0, \quad i=1,2,...,M $$

- 计算 \\(\frac{\partial L}{\partial w}, \ \frac{\partial L}{\partial b}, \ \frac{\partial L}{\partial \xi\_{i}}\\) 并令偏导数为 \\(0\\) 可得：

	$$ w = \sum\_{i=1}^{M} \alpha\_{i} y\_{i} x\_{i} \qquad \sum\_{i=1}^{M} \alpha\_{i} y\_{i} = 0 \qquad \alpha\_{i} + \beta\_{i} = C $$
	
	- 将上述结果代入 \\(L(w,b,\alpha)\\) 可得：

		$$ L(w,b,\alpha) = \sum\_{i=1}^{M} \alpha\_{i} - \frac{1}{2} \sum\_{i=1}^{M} \sum\_{j=1}^{M} \alpha\_{i} \alpha\_{j} \cdot y\_{i} y\_{j} \cdot x\_{i}^{T} x\_{j} $$

- 根据拉格朗日乘子法对偶性，上述问题可转化为：

	$$ \max\_{\alpha} G(\alpha) = \max\_{\alpha} \left( \sum\_{i=1}^{M} \alpha\_{i} - \frac{1}{2} \sum\_{i=1}^{M} \sum\_{j=1}^{M} \alpha\_{i} \alpha\_{j} \cdot y\_{i} y\_{j} \cdot x\_{i}^{T} x\_{j} \right) $$

	- 约束条件为：

		$$ 0 \leq \alpha\_{i} \leq C, \quad i=1,2,...,M $$

		$$ \sum\_{i=1}^{M} \alpha\_{i} y\_{i} = 0, \quad i=1,2,...,M $$

### 优化求解

#### 求解 \\(\alpha\\)

- 使用 SMO 算法求解

- 优化目标：\\(G(\alpha)\\)，每次固定其他变量，只更新 \\(\alpha\_{i}, \alpha\_{j}\\)，直到收敛

- 由 \\(\sum \alpha\_{i} y\_{i} = 0\\) 可得：

	$$ \alpha\_{i} y\_{i} + \alpha\_{i} y\_{j} = -\sum\_{k \neq i \neq j}^{M} \alpha\_{k} y\_{k} = D $$

	- 由 \\(y\_{i} \in \\{-1,+1\\}\\) 可得 \\(y\_{i} = \frac{1}{y\_{i}}\\)，上式可进一步变为：

		$$ \alpha\_{j} = (D-\alpha\_{i}y\_{i}) \cdot y\_{j} $$

- 固定其他变量，将 \\(\alpha\_{j}\\) 代入 \\(G(\alpha)\\) 可得：

	$$ G(\alpha) = A \alpha\_{i}^{2} + B \alpha\_{i} + E $$
	
- 由边界条件 \\(0 \leq \alpha\_{i}, \alpha\_{j} \leq C\\) 可知，该问题为区间内的二次函数最值问题，容易求解

#### 求解 \\(w\\)

- 求解出最优判别面的 \\(\alpha\_{i}\\) 后，代入下式计算 \\(w\\)：

	$$ w = \sum\_{i=1}^{M} \alpha\_{i} y\_{i} x\_{i} $$

#### 求解 \\(b\\)

- 假设 \\(V\\) 为支持向量的集合，对于支持向量 \\(x\_{v}\\)：

	$$ y\_{v} \left( \sum\_{i=1}^{M} \alpha\_{i} y\_{i} x\_{i}^{T} x\_{v} + b \right) = 1 $$

- 从上式可以求得：

	$$ b = \frac{1}{y\_{v}} - \sum\_{i=1}^{M} \alpha\_{i} y\_{i} x\_{i}^{T} x\_{v} = y\_{v} - \sum\_{i=1}^{M} \alpha\_{i} y\_{i} x\_{i}^{T} x\_{v} $$

- 通常在整个支持向量集合上求解 \\(b\\)：

	$$ b = \frac{1}{|V|} \sum\_{v \in V} \left( y\_{v} - \sum\_{i=1}^{M} \alpha\_{i} y\_{i} x\_{i}^{T} x\_{v} \right) $$

### 支持向量

- 将 \\(w, b\\) 代入 \\(h(x)\\) 可得最终判别面：

	$$ h(x) = \sum\_{i=1}^{M} \alpha\_{i} y\_{i} x\_{i}^{T} x + \frac{1}{|V|} \sum\_{v \in V} \left( y\_{v} - \sum\_{i=1}^{M} \alpha\_{i} y\_{i} x\_{i}^{T} x\_{v} \right) $$

#### 线性可分

- 由约束条件 \\(\alpha\_{i} \cdot (1 - y\_{i} \cdot (w^{T}x\_{i}+b)) = 0\\) 和 \\(\alpha\_{i} \geq 0\\) 可知：

	- 当 \\(\alpha\_{i} = 0\\) 时，样本 \\((x\_{i}, y\_{i})\\) 对 \\(h(x)\\) 无影响

	- 当 \\(\alpha\_{i} > 0\\) 时，\\(y\_{i} \cdot (w^{T}x\_{i}+b) = 1\\)，\\((x\_{i}, y\_{i})\\) 为支持向量

#### 线性不可分

- 由约束条件 \\(\alpha\_{i} \cdot (1 - \xi\_{i} - y\_{i} \cdot (w^{T}x\_{i}+b)) = 0\\) 和 \\(0 \leq \alpha\_{i} \leq C\\) 可知：

	- 当 \\(\alpha\_{i} = 0\\) 时，样本 \\((x\_{i}, y\_{i})\\) 对 \\(h(x)\\) 无影响

	- 当 \\(0 < \alpha\_{i} \leq C\\) 时，\\(y\_{i} \cdot (w^{T}x\_{i}+b) = 1 - \xi\_{i}\\)，\\((x\_{i}, y\_{i})\\) 为支持向量：

- 当 \\(0 < \alpha\_{i} \leq C\\) 时，由约束条件 \\(\beta\_{i} \cdot \xi\_{i} = 0\\) 和 \\( 0 \leq \beta\_{i} \leq C\\) 可知：

	- 当 \\(0 < \alpha\_{i} < C \\) 时， \\(\beta\_{i} > 0 \ \rightarrow \ \xi\_{i} = 0\\)，样本 \\((x\_{i},y\_{i})\\) 在最大间隔边界上

	- 当 \\(\alpha\_{i} = C \\) 时，\\(\beta\_{i} = 0 \ \rightarrow \ \xi\_{i} > 0\\)：

		- 若 \\(\xi\_{i} \leq 1\\)，样本 \\((x\_{i}, y\_{i})\\) 在最大间隔内部

		- 若 \\(\xi\_{i} > 1\\)，样本 \\((x\_{i}, y\_{i})\\) 分类错误

#### 判别面

- 综上所述，判别面只与支持向量有关，最终表达式为：

	$$ h(x) = \sum\_{v \in V} \alpha\_{v} y\_{v} x\_{v}^{T} x + \frac{1}{|V|} \sum\_{v \in V} \left( y\_{v} - \sum\_{s \in V} \alpha\_{s} y\_{s} x\_{s}^{T} x\_{v} \right) $$