<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 最大熵

## 基本思想

- 在所有可能的概率模型中，熵最大的模型是最好的模型

- 当随机变量 \\(X\\) 服从均匀分布时，熵最大为 \\(log|X|\\)

- 对偶函数的极大化，等价于最大熵模型的极大似然估计

## 数学推导

### 模型定义

- 联合分布的经验分布计算如下：

	$$ \tilde{P\ }(X=x, Y=y) = \frac{C(X=x, Y=y)}{M} $$
	
	- 其中，\\(C(\cdot)\\) 表示训练集中满足条件 \\(\cdot\\) 的样本出现次数

- 边缘分布的经验分布计算如下：

	$$ \tilde{P\ }(X=x) = \frac{C(X=x)}{M} $$

- 特征函数定义如下：

	- 当 \\(x, \ y\\) 满足某一事实时，特征函数取值为 \\(f(x,y)=1\\)

	- 当 \\(x, \ y\\) 不满足某一事实时，特征函数取值为 \\(f(x,y)=0\\)

- 特征函数 \\(f(x,y)\\) 关于经验分布 \\(\tilde{P\ }(X,Y)\\) 的期望 \\(E\_{\tilde{P\ }}(f)\\) 计算如下：

	$$ E\_{\tilde{P\ }}(f) = \sum\_{x,y} \tilde{P\ }(x,y) \cdot f(x,y) $$

- 特征函数 \\(f(x,y)\\) 关于模型 \\(P(Y|X)\\) 与经验分布 \\(\tilde{P\ }(X)\\) 的期望 \\(E\_{P\ }(f)\\) 计算如下：

	$$ E\_{P}(f) = \sum\_{x,y} \tilde{P\ }(x) \cdot P(y|x) \cdot f(x,y) $$

- 如果模型能够获取训练数据中的信息，可以假设这两个期望值相等：

	$$ E\_{\tilde{P\ }}(f) = E\_{P}(f) $$

- 定义在条件概率 \\(P(Y|X)\\) 上的条件熵为：

	$$ H(P) = -\sum\_{x,y} \tilde{P\ }(x) \cdot P(y|x) \cdot log P(y|x) $$

### 参数学习

- 优化目标：

	$$ \min\_{P(Y|X)} \sum\_{x,y} \tilde{P\ }(x) \cdot P(y|x) \cdot log P(y|x) $$

	- 约束条件：

		$$ E\_{\tilde{P\ }}(f\_{i}) = E\_{P}(f\_{i}) \qquad i=1, \ 2, \ \cdots, \ n $$
		
		$$ \sum\_{y} P(y|x) = 1 $$

- 由拉格朗日乘子法：

	$$ L(P,w) = \tilde{P\ }(x) \cdot P(y|x) \cdot log P(y|x) + w\_{0} \left( 1 - \sum\_{y} P(y|x) \right) + \sum\_{i=1}^{n} w\_{i} \left( \tilde{P\ }(x,y) \cdot f\_{i}(x,y) - \tilde{P\ }(x) \cdot P(y|x) \cdot f\_{i}(x,y) \right) $$

- 对 \\(P(y|x)\\) 求导：

	$$ \frac{\partial{L(P,w)}}{\partial(P(y|x))} = \tilde{P\ }(x) \left( log P(y|x) + 1 - w\_{0} - \sum\_{i=1}^{n} w\_{i} \cdot f\_{i}(x,y) \right) $$

- 在 \\(\tilde{P\ } > 0\\) 前提下，令导数为 \\(0\\) 可得：

	$$ P(y|x) = \frac{exp \left( \sum\_{i=1}^{n} w\_{i} \cdot f\_{i}(x,y) \right)}{exp(1-w\_{0})} $$

- 由全概率公式 \\(\sum\_{y} P(y|x)=1\\) 可得：

	$$ exp(1-w\_{0}) = \sum\_{y} exp \left( \sum\_{i=1}^{n} w\_{i} \cdot f\_{i}(x,y) \right) $$

- 因此最终条件概率计算如下：

	$$ P\_{w}(y|x) = \frac{1}{Z\_{w}(x)} exp \left( \sum\_{i=1}^{n} w\_{i} \cdot f\_{i}(x,y) \right) $$

	- \\(Z\_{w}(x)\\) 是归一化参数：

		$$ Z\_{w}(x) = \sum\_{y} exp \left( \sum\_{i=1}^{n} w\_{i} \cdot f\_{i}(x,y) \right) $$

	- \\(f\_{i}(x,y)\\) 是第 \\(i\\) 个特征函数，\\(w\_{i}\\) 是其对应权值

- 将 \\(P\_{w}(y|x)\\) 代入 \\(L(p,w)\\) 并求解 \\(\arg \max\_{w} L(P\_{w},w)\\) 即可得到 \\(w^{\*}\\)，进而求出 \\(P\_{w}^{\*}(y|x)\\)