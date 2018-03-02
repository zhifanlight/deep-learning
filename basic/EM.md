<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# EM 算法

## 基本思想

- 给定 \\(m\\) 个样本 \\(\\{(x\_{1}, z\_{1}), \ (x\_{2}, z\_{2}), \ \cdots, \ (x\_{m}, z\_{m})\\}\\)，假设样本之间相互独立

- 想要拟合模型 \\(P(x,z)\\) 的参数 \\(\theta\\)，但只有 \\(x\\) 可见，而隐变量 \\(z\\) 不可见

- 交叉进行 E 步和 M 步：

	- E 步：计算隐变量 \\(z\\) 的后验分布 \\(Q(z)\\)

	- M 步：根据当前隐变量后验概率，更新模型参数

- 由后续推导可知，M 步通过最大化对数似然的下界，来逼近对数似然函数

## 数学推导

- 对数似然函数如下：

	$$ L(\theta) = \sum\_{i=1}^{m} log \ P(x\_{i} \ ; \theta) = \sum\_{i=1}^{m} log \left( \sum\_{z\_{i}} P(x\_{i}, z\_{i} \ ; \theta)\right) $$

- 假设 \\(Q\_{i}(z\_{i})\\) 是概率分布，\\(L(\theta)\\) 可推导为：

	$$ L(\theta) = \sum\_{i=1}^{m} log \left( \sum\_{z\_{i}} Q\_{i}(z\_{i}) \cdot \frac{P(x\_{i}, z\_{i} \ ; \theta)}{Q\_{i}(z\_{i})} \right) = \sum\_{i=1}^{m} log \ E \left[ \frac{P(x\_{i}, z\_{i} \ ; \theta)}{Q\_{i}(z\_{i})} \right] $$

- 由于对数函数 \\(log\\) 为凹函数，由 Jensen 不等式：

	$$ L(\theta) \geq \sum\_{i=1}^{m} E \left[ log \left( \frac{P(x\_{i}, z\_{i} \ ; \theta)}{Q\_{i}(z\_{i})} \right) \right] = \sum\_{i=1}^{m} \sum\_{z\_{i}} Q\_{i}(z\_{i}) \cdot log \ \frac{P(x\_{i}, z\_{i} \ ; \theta)}{Q\_{i}(z\_{i})} $$

	- 等号成立当且仅当 \\(\forall \ z\_{i} \rightarrow \frac{P(x\_{i}, z\_{i} \ ; \theta)}{Q\_{i}(z\_{i})} \equiv C\\)

- 由概率之和 \\(\sum\_{z\_{i}} Q\_{i}(z\_{i}) = 1\\)：

	$$ \sum\_{z\_{i}} \frac{1}{C} P(x\_{i}, z\_{i} \ ; \theta) = \frac{1}{C} P(x\_{i} \ ; \theta) = 1 $$

- 由前一步推导结果：

	$$ Q\_{i}(z\_{i}) = \frac{P(x\_{i}, z\_{i} \ ; \theta)}{C} = \frac{P(x\_{i} \ ; \theta)}{C} \cdot P(z\_{i}|x\_{i} \ ; \theta) = P(z\_{i}|x\_{i} \ ; \theta) $$

## 一般形式

### E 步

$$ Q\_{i}(z\_{i}) = P(z\_{i}|x\_{i} \ ; \theta) $$

### M 步

$$ \theta = \begin{equation} \mathop{\arg\max}_{\theta} \sum\_{i=1}^{m} \sum\_{z\_{i}} \left( Q\_{i}(z\_{i}) \cdot log \frac{P(x\_{i}, z\_{i} \ ; \theta)}{Q\_{i}(z\_{i})} \right) \end{equation} $$