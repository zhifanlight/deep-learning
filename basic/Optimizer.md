<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 优化方法

## 背景介绍

- 建立模型后，需要最小化损失函数

- 一般采用基于梯度的方法及其变种，对参数进行持续迭代更新，直到模型趋于稳定

## 常用优化方法

### Gradient Descent Optimizer

- 通常所说的梯度下降，是指 Mini-batch Gradient Descent

- 更新公式均为 \\( \theta \leftarrow \theta - \eta \nabla\_{\theta} J(\theta) \\)，但计算梯度时使用的样本数量不同

#### Batch Gradient Descent

- 更新参数时，使用所有的样本计算梯度

- 优点：

	- 能得到全局最优解，迭代次数少

- 缺点：

	- 当样本集太大时，训练过程缓慢

#### Stochastic Gradient Descent

- 更新参数时，使用单个样本计算梯度

- 优点：

	- 训练速度快

- 缺点：

	- 训练过程不稳定，得不到全局最优解

#### Mini-batch Gradient Descent

- 更新参数时，使用 N 个样本计算梯度；N 一般为 50 ～ 256

- 优点：

	- 训练速度较快，接近全局最优解

#### 总结

- 选择合适的学习率 \\(\eta\\) 非常困难

- 当一个方向上的梯度变化比另一个方向陡峭，训练过程会发生明显振荡

- 对每个参数的学习率都相同，这种做法不合理：

	- 对于稀疏数据，如果特征出现频率低，应该设置较大的学习率

	- 对于稀疏数据，如果特征出现频率高，应该设置较小的学习率

- 鞍点附近梯度较小，如果鞍点较多，会导致训练过程接近停滞

### Momentum Optimizer

- 借用物理中动量的概念，模拟物体运动时的惯性，在一定程度上保留之前的更新方向，同时根据当前 Batch 的梯度微调最终的更新方向

#### Momentum

- 参数更新过程如下：

	$$ v\_{t} \leftarrow \gamma v\_{t-1} + \eta \nabla\_{\theta} J(\theta) $$
	
	$$ \theta \leftarrow \theta - v\_{t} $$
	
	- 超参数 \\(\gamma\\) 通常设置为 0.9

- 优点：

	- 由于保留了之前的更新方向，在一定程度上能够减少振荡，从而加速收敛

#### Nesterov

- 用 \\(\theta - \gamma v\_{t-1}\\) 近似下一步更新后的参数值，通过未来参数 \\(\theta\\) 的梯度更新当前的参数：

	$$ v\_{t} \leftarrow \gamma v\_{t-1} + \eta \nabla\_{\theta} J(\theta - \gamma v\_{t-1}) $$
	
	$$ \theta \leftarrow \theta - v\_{t} $$
	
	- 超参数 \\(\gamma\\) 通常设置为 0.9

- 优点：

	- 具有预见性的更新，可以防止前进的太快，提高灵敏度

	- 对一些 RNN 任务，性能提升明显

### Adagrad Optimizer

- 让学习率适应参数，适合处理稀疏数据：

	- 对于出现次数少的特征，采用较大的学习率

	- 对于出现次数多的特征，采用较小的学习率

- 令 \\(g\_{t,i}\\) 表示 \\(t\\) 时刻参数 \\(\theta\_{i}\\) 的梯度，参数更新如下：

	$$ \theta\_{t+1,i} \leftarrow \theta\_{t,i} - \frac{\eta}{\sqrt{G\_{t,ii} + \epsilon}} \cdot g\_{t,i} $$
		
	- 对角矩阵 \\(G\_{t}\\) 的每个元素 \\(G\_{t,ii}\\) 表示到 \\(t\\) 时刻为止，关于 \\(\theta\_{i}\\) 的梯度平方和

	- \\(\epsilon\\) 是平滑项，防止分母为 0，通常取 \\(1e^{-8}\\)
	
- 通过对位乘法进行向量化：

	$$ \theta\_{t+1} \leftarrow \theta\_{t} - \frac{\eta}{\sqrt{G\_{t} + \epsilon}} \odot g\_{t} $$

- 优点：

	- 无需手动调节，只需设置一个初始学习率，通常设置为 0.01

- 缺点：

	- 学习率调整过于激进，分母不断累积导致学习率不断收缩，训练过程可能提前结束

### Adadelta Optimizer

- Adagrad 方法的扩展算法，处理 Adagrad 学习率单调递减的问题

- 不使用 \\(\theta\\) 的梯度平方和，而是递归地计算梯度平方的历史均值：

	$$ E \left[ g^{2} \right]\_{t} \leftarrow \gamma E \left[ g^{2} \right]\_{t-1} + (1 - \gamma) g\_{t}^{2} $$

	- 超参数 \\(\gamma\\) 通常设置为 0.9

- 用 \\(E \left[ g^{2} \right]\_{t}\\) 代替 Adagrad 中的 \\(G\_{t}\\)，参数更新如下：

	$$ \Delta \theta\_{t} \leftarrow - \frac{\eta}{\sqrt{E \left[ g^{2} \right]\_{t} + \epsilon}} \ g\_{t} $$

- 进一步近似可得：

	$$ \Delta \theta\_{t} \leftarrow - \frac{RMS \left[ \Delta \theta \right]\_{t-1} }{RMS \left[ g \right]\_{t} } \ g\_{t} $$
	
	$$ \theta\_{t+1} \leftarrow \theta\_{t} + \Delta \theta\_{t} $$

	- \\(RMS \left[ g \right]\_{t} = \sqrt{E \left[ g^{2} \right]\_{t} + \epsilon }\\) 是 \\(t\\) 时刻 \\(g^{2}\\) 历史均值的平方根
	
	- \\(RMS \left[ \Delta \theta \right]\_{t} = \sqrt{E \left[ \Delta \theta^{2} \right]\_{t} + \epsilon }\\) 是 \\(t\\) 时刻 \\(\Delta \theta^{2}\\) 历史均值的平方根：

		$$ E \left[ \Delta \theta ^{2} \right]\_{t} \leftarrow \gamma E \left[ \Delta \theta^{2} \right]\_{t-1} + (1 - \gamma) \Delta \theta\_{t}^{2} $$

- 优点：

	- 无需设置默认学习率，训练初期加速效果不错

- 缺点：

	- 训练后期，容易在局部最小值附近抖动

### RMSProp Optimizer

- Adadelta 的一个特例

- 不使用 \\(\theta\\) 的梯度平方和，而是递归地计算梯度平方的历史均值：

	$$ E \left[ g^{2} \right]\_{t} \leftarrow \gamma E \left[ g^{2} \right]\_{t-1} + (1 - \gamma) g\_{t}^{2} $$

	- 超参数 \\(\gamma\\) 通常设置为 0.9

- 参数更新如下：

	$$ \theta\_{t+1} \leftarrow \theta\_{t} - \frac{\eta}{\sqrt{E \left[ g^{2} \right]\_{t} + \epsilon}} \ g\_{t} $$
	
	- 超参数 \\(\eta\\) 通常设置为 0.001

### Adam Optimizer

- 本质上是带偏差校正和动量项的 RMSProp

- 像 Adadelta 和 RMSProp 一样，保存梯度平方的历史均值 \\(v\_{t}\\)；除此之外，还保存梯度的历史均值 \\(m\_{t}\\)，类似于动量：

	$$ \left\\{ \begin{matrix} m\_{t} \leftarrow \beta\_{1} \ m\_{t-1} + (1 - \beta\_{1}) \ g\_{t} \\\\ v\_{t} \leftarrow \beta\_{2} \ v\_{t-1} + (1 - \beta\_{2}) \ g\_{t}^{2} \end{matrix} \right. $$
	
	- \\(m\_{t}, v\_{t}\\) 分别是对梯度的一阶矩（均值）、二阶矩（方差）的估计

	- 超参数 \\(\beta\_{1}, \beta\_{2}\\) 通常设置为 0.9，0.999

- \\(m\_{t}, v\_{t}\\)容易偏向于 0，需要进行偏置校正：

	$$ \left\\{ \begin{matrix} \tilde{m\_{t}} = \frac{m\_{t}}{1 - \beta\_{1}^{t}} \\\\ \tilde{v\_{t}} = \frac{v\_{t}}{1 - \beta\_{2}^{t}} \end{matrix} \right. $$
	
- 最终参数更新如下：

	$$ \theta\_{t+1} \leftarrow \theta\_{t} - \frac{\eta}{\sqrt{\tilde{v\_{t}}} + \epsilon} \ \tilde{m\_{t}} $$
	
	- 超参数 \\(\eta\\) 通常设置为 0.001

- 优点：

	- 由于优化后期梯度越来越稀疏，偏差校正使得 Adam 在实际中表现更好

## 如何选择

- 如果数据是稀疏的，就用自适用方法，即 Adagrad，Adadelta，RMSprop，Adam

- RMSprop，Adadelta，Adam 在很多情况下的效果相似

- 随着梯度变得稀疏，Adam 比 RMSprop 效果会好；整体来讲，Adam 是最好的选择

![img](images/optimizer_3d.gif)

![img](images/optimizer_2d.gif)