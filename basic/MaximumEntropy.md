<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 最大熵

## 基本思想

- 在所有可能的概率模型中，熵最大的模型是最好的模型

- 用于二分类时，可以推导出 sigmoid 函数和逻辑回归

## 数学推导

### 模型定义

- 联合分布的经验分布计算如下：

	$$ \tilde{P\ }(X=x, Y=y) = \frac{C(X=x, Y=y)}{M} $$
	
	- 其中，\\(C(\cdot)\\) 表示训练集中满足条件 \\(\cdot\\) 的样本出现次数

- 边缘分布的经验分布计算如下：

	$$ \tilde{P\ }(X=x) = \frac{C(X=x)}{M} $$

- 特征函数描述输入 \\(x\\) 和输出 \\(y\\) 之间的某个事实，定义如下：

	- 当 \\(x, \ y\\) 满足某一事实时，特征函数取值为 \\(f(x,y)=1\\)

	- 当 \\(x, \ y\\) 不满足某一事实时，特征函数取值为 \\(f(x,y)=0\\)

- 特征函数 \\(f(x,y)\\) 在训练集上关于 \\(\tilde{P\ }(X,Y)\\) 的期望 \\(E\_{\tilde{P\ }}(f)\\) 计算如下：

	$$ E\_{\tilde{P\ }}(f) = \sum\_{x,y} \tilde{P\ }(x,y) \cdot f(x,y) $$

- 特征函数 \\(f(x,y)\\) 在模型上关于 \\(P(X,Y)\\) 的期望 \\(E\_{P}(f)\\) 计算如下：

	$$ E\_{P}(f) = \sum\_{x,y} P(x,y) \cdot f(x,y) = \sum\_{x,y} P(x) \cdot P(y|x) \cdot f(x,y) $$

	- 由于 \\(P(x)\\) 不易计算，通常用 \\(\tilde{P\ }(x)\\) 近似：

		$$ E\_{P}(f) = \sum\_{x,y} \tilde{P\ }(x) \cdot P(y|x) \cdot f(x,y) $$

- 对于概率分布 \\(P(y|x)\\)，希望特征 \\(f\\) 的期望值和训练集上的特征期望值相同：

	$$ E\_{P}(f) = E\_{\tilde{P\ }}(f) $$
	
	- 即：

		$$ \sum\_{x,y} \tilde{P\ }(x) \cdot P(y|x) \cdot f(x,y) = \sum\_{x,y} \tilde{P\ }(x,y) \cdot f(x,y) $$

- 定义在条件概率 \\(P(Y|X)\\) 上的条件熵为：

	$$ H(Y|X) = -\sum\_{x,y} \tilde{P\ }(x) \cdot P(y|x) \cdot \log P(y|x) $$
	
	- 同理，\\(\tilde{P\ }(x)\\) 用来近似 \\(P(x)\\)

### 参数学习

- 优化目标：

	$$ \min\_{P(Y|X)} \sum\_{x,y} \tilde{P\ }(x) \cdot P(y|x) \cdot \log P(y|x) $$

	- 约束条件：

		$$ E\_{\tilde{P\ }}(f\_{i}) = E\_{P}(f\_{i}) \qquad i=1, \ 2, \ \cdots, \ n $$
		
		$$ \sum\_{y} P(y|x) = 1 $$

- 由拉格朗日乘子法：

	$$ L(P,w) = \sum\_{x,y} \tilde{P\ }(x) \cdot P(y|x) \cdot \log P(y|x) + w\_{0} \left( 1 - \sum\_{y} P(y|x) \right) + \sum\_{i=1}^{n} w\_{i} \left( \sum\_{x,y} \tilde{P\ }(x,y) \cdot f\_{i}(x,y) - \sum\_{x,y} \tilde{P\ }(x) \cdot P(y|x) \cdot f\_{i}(x,y) \right) $$

- 最终条件概率计算如下：

	$$ P\_{w}(y|x) = \frac{1}{Z\_{w}(x)} \exp \left( \sum\_{i=1}^{n} w\_{i} \cdot f\_{i}(x,y) \right) $$

	- 其中 \\(Z\_{w}(x)\\) 是归一化参数：

		$$ Z\_{w}(x) = \sum\_{y} \exp \left( \sum\_{i=1}^{n} w\_{i} \cdot f\_{i}(x,y) \right) $$

	- \\(f\_{i}(x,y)\\) 是第 \\(i\\) 个特征函数，\\(w\_{i}\\) 是对应权重

- 代入 \\(L(p,w)\\) 并求解 \\(\arg \max\_{w} L(P\_{w},w)\\) 即可得到 \\(w^{\*}\\)，进而求出 \\(P\_{w}^{\*}(y|x)\\)