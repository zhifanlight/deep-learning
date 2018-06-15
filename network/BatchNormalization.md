<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Batch Normalization

## 背景介绍

### Covariate Shift

- 网络浅层参数的改变，导致深层的输入分布发生变化，是神经网络训练困难的原因

### Whitening

- 对输入进行白化（零均值、单位标准差）能够加速网络收敛

- 对于 Sigmoid 激活函数，可以避免输入层的梯度消失，但深层数据分布依旧不可控

### Batch Normalization

- 核心思想是通过对中间层数据进行归一化和特征恢复，减少浅层网络输出分布变化对深层网络输入的影响，加速训练

## 数学推导

### 训练过程

- 计算输入 \\(x\\) 的均值 \\(\mu\_{B}\\) 与方差 \\(\sigma\_{B}^{2}\\)：

	$$ \mu\_{B} = \frac{1}{m} \sum\_{i=1}^{m} x\_{i} \qquad \sigma\_{B}^{2} = \frac{1}{m} \sum\_{i=1}^{m} (x\_{i} - \mu\_{B})^{2} $$

- 将输入 \\(x\\) 归一化为零均值、单位方差的 \\(\hat{x\_{i}}\\)：

	$$ \hat{x\_{i}} = \frac{x\_{i} - \mu\_{B}}{\sqrt{\sigma\_{B}^{2} + \epsilon}} $$

- 由于 \\(\hat{x\_{i}}\\) 改变了输入的分布，通过两个可学习的参数还原输入分布：

	$$ y\_{i} = \gamma \hat{x\_{i}} + \beta $$

	- \\(y\_{i}\\) 即为 Batch Normalization 后的输出

	- 训练稳定时，\\(\gamma\\) 近似等于 \\(x\_{i}\\) 的标准差 \\(\sigma\_{B}\\)，\\(\beta\\) 近似等于 \\(x\_{i}\\) 的均值 \\(\mu\_{B}\\)

	- 由于 \\(\hat{x\_{i}}\\) 分布固定， \\(y\_{i}\\) 的分布不再受浅层网络参数的影响，可以加速训练过程

	- \\(\hat{x\_{i}}\\) 被归一化为标准正态分布，直接作为下一层输入会限制网络表达能力；需要通过参数 \\(\gamma, \beta\\) 进行恢复，具体恢复程度在训练过程中由神经网络自主决定

### 测试过程

- Batch Normalization 是为了方便训练，测试时不应该使用；但由于训练时作为网络的整体，直接拿掉会影响结果

- 首先计算训练过程中所有的 batch 的均值和方差：

	$$ \mu = E \left\[\mu\_{B}\right\] \qquad \sigma^{2} = \frac{m}{m-1} E [\sigma\_{B}^{2}] $$

- 此时 Batch Normalization 的输出值为：

	$$ y = \gamma \left( \frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}} \right) + \beta = \frac{\gamma}{\sqrt{\sigma^{2}+\epsilon}} \cdot x + \left( \beta - \frac{\gamma \mu}{\sqrt{\sigma^{2}+\epsilon}}\right) $$

	- 由于 \\(\gamma\\) 是 \\(\sigma\\) 的近似，\\(\beta\\) 是 \\(\mu\\) 的近似，上式近似等于：

		$$ y = x $$
			
		- 相当于不使用 Batch Normalization

- 在 CNN 中，通常为每个特征图设置一个 \\(\gamma, \ \beta\\)，以减少参数量

### 偏置项

- 对于使用 Batch Normalization 的层，无需使用偏置项 \\(b\\)：

	- Batch Normalization 层的 \\(\beta\\) 会抵消掉 \\(b\\) 的影响

		$$ BN(w^{T}x) = BN(w^{T}x + b) $$

### 激活层

- Batch Normalization 层通常放在卷积层之后、激活层之前：

	- \\(w^{T}x+b\\) 更接近高斯分布，Batch Normalization 的归一化效果更好

	- 如果放在激活层之后，\\(g(w^{T}x+b)\\) 则不具有这一特性

## 性能分析

### 加速训练

- 解决 Covariate Shift：

	- 对特征进行归一化，通过两个可学习的参数 \\(\gamma, \ \beta\\) 进行恢复，可以避免参数更新后浅层输出对深层输入的影响

- 直接使用较高的学习率：

	- 如果每一层、每一维的数据范围不同，其需要的学习率也不同，通常使用最小的学习率才能保证损失函数下降

	- Batch Normalization 将每一层、每一维的数据范围保持一致，可以直接使用较高的学习率进行训练

### 解决梯度问题

- 对于 \\(x\_{l} = w^{T}\_{l} x\_{l-1}\\)，在反向传播时计算如下：

	$$ \frac{\partial{l}}{\partial{x\_{l-1}}} = \frac{\partial{l}}{\partial{x\_{l}}} \cdot \frac{\partial{x\_{l}}}{\partial{x\_{l-1}}} = \frac{\partial{l}}{\partial{x\_{l}}} w\_{l} $$

	- 从第 \\(l\\) 层传到第 \\(k\\) 层时计算如下：

		$$ \frac{\partial{l}}{\partial{x\_{k}}} = \frac{\partial{l}}{\partial{x\_{l}}} \prod\_{i=k+1}^{l} w\_{i} $$

	- 如果大部分 \\(w\_{i} > 1\\)，会导致梯度爆炸

	- 如果大部分 \\(w\_{i} < 1\\)，会导致梯度消失

- 由 Batch Normalization 过程可得：

	$$ x\_{l} = BN(w^{T}\_{l}x\_{l-1}) = BN(aw^{T}\_{l}x\_{l-1}) $$

	- 反向传播时求导如下：

		$$ \frac{\partial{x\_{l}}}{\partial{x\_{l-1}}} = \frac{\partial{BN(w^{T}\_{l}x\_{l-1})}}{\partial{x\_{l-1}}} =  \frac{\partial{BN(\alpha w^{T}\_{l}x\_{l-1})}}{\partial{x\_{l-1}}} $$

	- 反向传播时的残差与 \\(w\\) 的尺度无关：

		- 尽管在参数更新时改变了 \\(w\\) 的值，但反向传播时的残差却不受影响

- 由梯度下降法可得：
	
	$$ \frac{\partial{l}}{\partial{(\alpha w\_{l})}} = \frac{\partial{l}}{\partial{x\_{l}}} \cdot \frac{\partial{x\_{l}}}{\partial{(\alpha w\_{l})}} = \frac{\partial{l}}{\partial{x\_{l}}} \cdot \left( \frac{1}{\alpha} \cdot \frac{\partial{BN(w\_{l}^{T}x\_{l-1})}}{\partial{w\_{l}}} \right) $$

	- 当 \\(w\\) 增大时，梯度减小，同样的学习率下参数更新幅度较小，可以防止梯度爆炸

### 防止过拟合

- 由 Batch Normalization 过程可知，某个样本的中间层特征不再仅取决于样本本身，也取决于这个样本所属的 mini-batch

- 同一个样本与不同的样本组成 mini-batch 时，输出也不同，一定程度上可以抑制过拟合