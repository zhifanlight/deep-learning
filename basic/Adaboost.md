<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Adaboost

## 背景介绍

- 对于分类问题，训练弱分类器要比强分类器容易

- 从弱分类器出发，反复学习，得到一系列弱分类器，将这些弱分类器组合成一个强分类器

- Adaboost 提高错误分类样本的权重，降低正确分类样本的权重，权值的变动将使新的分类器更关注之前被错误分类的样本

- Adaboost 损失函数按指数速率下降

## 学习过程

- 假设样本集为 \\(\\{(x\_{1},y\_{1}), (x\_{2},y\_{2}), \cdots, (x\_{M},y\_{M})\\}\\)，样本标签 \\(y\_{i} \in \\{-1, +1\\} \\)，样本的初始权重 \\(W\_{1}=(\frac{1}{M}, \frac{1}{M}, \cdots, \frac{1}{M})\\)

- 指示函数定义如下：

	$$ I(c)=\\left\\{ \begin{matrix} 1, \quad if \ c \ is \ true \\\\ 0, \quad if \ c \ is \ false \end{matrix} \\right. $$

- 进行 \\(T\\) 次迭代，对于当前迭代次数 \\(t\\)：

	- 根据当前权重 \\(W\_{t}\\) 训练弱分类器 \\(h\_{t}(x)\\)，并计算错误率 \\(e\_{t}\\)：

		$$ e\_{t} = \sum\_{i=1}^{M} W\_{t,i} \cdot I(h\_{t}(x\_{i}) \neq y\_{i}) $$

	- 若 \\(e\_{t} > 0.5\\)，结束迭代

	- 计算分类器 \\(h\_{t}(x)\\) 的权重 \\(\alpha\_{t}\\)：

		$$ \alpha\_{t} = \frac{1}{2} \ln \left( \frac{1-e\_{t}}{e\_{t}} \right) $$
	
	- 更新样本权重：
	
		$$ W\_{t+1,i} = \frac{W\_{t,i}}{Z\_{t}} \cdot e^{-\alpha\_{t} \cdot y\_{i} \cdot h\_{t}(x\_{i})} $$
		
		- 其中 \\(Z\_{t}\\) 是归一化系数：

			$$ Z\_{t} = \sum\_{i=1}^{M} W\_{t,i} \cdot e^{-\alpha\_{t} \cdot y\_{i} \cdot h\_{t}(x\_{i})} $$

- 得到最终的强分类器 \\(H(x)\\)：

	$$ H(x) = sign \left( \sum\_{t=1}^{T} \alpha\_{t} \cdot h\_{t}(x) \right) $$

## 数学推导

### 错误率 \\(e\_{t}<0.5\\)

- 如果弱分类器的性能还不如随机猜测，那么学到的模型将毫无意义

### 归一化系数 \\(Z\_{t}\\)

- 使 \\(W\_{t}\\) 成为一个概率分布

- 也是弱分类器 \\(h\_{t}(x)\\) 的损失函数

### 计算 \\(h\_{t}(x)\\) 权重 \\(\alpha\_{t}\\)

#### 误差上限

- 当 \\(H(x\_{i}) \neq y\_{i}\\) 时，\\(y\_{i} H(x\_{i}) = -1\\)，此时 \\(e^{-y\_{i}H(x\_{i})} > I(H(x\_{i}) \neq y\_{i})\\)，误差上限如下：

	$$ \frac{1}{M} \sum\_{i=1}^{M} I(H(x\_{i}) \neq y\_{i}) < \frac{1}{M} \sum\_{i=1}^{M} e^{-y\_{i}H(x\_{i})} $$

- 由每一轮样本权重更新公式可得：

	$$ Z\_{t} \cdot W\_{t+1,i} =W\_{t,i} \cdot e^{-\alpha\_{t} \cdot y\_{i} \cdot h\_{t}(x\_{i})} $$

- 将上式代入误差上限可得：

	$$
	\begin{aligned}
	\frac{1}{M} \sum\_{i=1}^{M} e^{-y\_{i}H(x\_{i})} &= \sum\_{i=1}^{M} W\_{1,i} \cdot \exp \left( \sum\_{t=1}^{T} -\alpha\_{t} \cdot y\_{i} \cdot h\_{t}(x\_{i}) \right) \newline
	&= \sum\_{i=1}^{M} W\_{1,i} \cdot \prod\_{t=1}^{T} \exp(-\alpha\_{t} \cdot y\_{i} \cdot h\_{t}(x\_{i})) \newline
	&= \sum\_{i=1}^{M} W\_{1,i} \cdot \exp(-\alpha\_{1} \cdot y\_{i} \cdot h\_{1}(x\_{i})) \cdot \prod\_{t=2}^{T} \exp(-\alpha\_{t} \cdot y\_{i} \cdot h\_{t}(x\_{i})) \newline
	&= Z\_{1} \cdot \sum\_{i=1}^{M} W\_{2,i} \cdot \prod\_{t=2}^{T} \exp(-\alpha\_{t} \cdot y\_{i} \cdot h\_{t}(x\_{i})) \newline
	&= \prod\_{t=1}^{T}Z\_{t} \cdot \sum\_{i=1}^{M} W\_{t+1,i} \newline
	&= \prod\_{t=1}^{T}Z\_{t} \newline
	\end{aligned}
	$$

- 因此，在每一轮训练弱分类器 \\(h\_{t}(x)\\) 时，应最小化归一化因子 \\(Z\_{t}\\)

#### 系数 \\(\alpha\_{t}\\)

- 对于每一个 \\(\alpha\_{t}\\)，都要最小化损失函数 \\(Z\_{t}\\)：

	$$
	\begin{aligned}
	Z\_{t} &= \sum\_{i=1}^{M} W\_{t,i} \cdot e^{-\alpha\_{t} \cdot y\_{i} \cdot h\_{t}(x\_{i})} \newline
	&= \sum\_{y\_{i}=h\_{t}(x\_{i})} W\_{t,i} \cdot e^{-\alpha\_{t}} + \sum\_{y\_{i} \neq h\_{t}(x\_{i})} W\_{t,i} \cdot e^{\alpha\_{t}} \newline
	&= e^{-\alpha\_{t}} \cdot \sum\_{y\_{i}=h\_{t}(x\_{i})} W\_{t,i} \cdot I(h\_{t}(x\_{i}) = y\_{i}) + e^{\alpha\_{t}} \cdot \sum\_{y\_{i} \neq h\_{t}(x\_{i})} W\_{t,i} \cdot I(h\_{t}(x\_{i}) \neq y\_{i}) \newline
	&= e^{-\alpha\_{t}} \cdot (1 - e\_{t}) + e^{\alpha\_{t}} \cdot e\_{t} \newline
	\end{aligned}
	$$

- 计算 \\(\frac{\partial Z\_{t}}{\partial \alpha\_{t}}\\) 并令偏导数为 \\(0\\) 可得：

	$$ \alpha\_{t} = \frac{1}{2} \ln \left( \frac{1-e\_{t}}{e\_{t}} \right) $$

#### 样本权重

- 将 \\(\alpha\_{t}\\) 代入 \\(Z\_{t}\\) 推导过程可得：

	$$ Z\_{t} = 2\sqrt{e\_{t} \cdot (1-e\_{t})} $$

- 对于正确分类的样本：

	$$ \frac{e^{-\alpha\_{t} \cdot y\_{i} \cdot h\_{t}(x\_{i})}}{Z\_{t}} = \frac{e^{-\alpha\_{t}}}{Z\_{t}} = \frac{1}{2 \cdot (1-e\_{t})} < 1 $$
	
	- 在下一次迭代时，样本权重会减小

- 对于错误分类的样本：

	$$ \frac{e^{-\alpha\_{t} \cdot y\_{i} \cdot h\_{t}(x\_{i})}}{Z\_{t}} = \frac{e^{\alpha\_{t}}}{Z\_{t}} = \frac{1}{2 \cdot e\_{t}} > 1 $$
	
	- 在下一次迭代时，样本权重会增大

#### 指数速率

- 令 \\(\gamma\_{t} = \left( \frac{1}{2} - e\_{t} \right) \in [0, 0.5)\\)，代入 \\(Z\_{t}\\) 可得：

	$$ Z\_{t} = \sqrt{1 - 4\gamma\_{t}^{2}} $$

- 由导数易知，\\(f(x) = e^{-4 x^{2}} - (1 - 4 x^{2}) \geq 0\\) 在 \\([0, 0.5)\\) 上恒成立，因此：

	$$ \sqrt{1 - 4\gamma\_{t}^{2}} \leq e^{-2 \gamma\_{t}^2} $$

- 将 \\(Z\_{t}\\) 与上述不等式代入误差上限可得：

	$$ \prod\_{t=1}^{T}Z\_{t} \leq \prod\_{t=1}^{T}e^{-2 \gamma\_{t}^2} \leq \exp \left( - \sum\_{t=1}^{T} 2 \gamma\_{t}^2 \right) $$

- 因此 Adaboost 误差上限较小，同时还以指数速率下降