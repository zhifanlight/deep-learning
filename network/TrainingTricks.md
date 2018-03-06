<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 训练技巧

## 数据增强

- 参考 [DataPreprocess.md](../basic/DataPreprocess.md)

## 正则化

- 通过 \\(L\_{1}\\) 或 \\(L\_{2}\\) 正则项，使参数大部分为 0 或尽量接近 0，一定程度上简化模型

- 关于正则化，参考 [Regularization.md](../basic/Regularization.md)

## Dropout

- 仅在训练时使用，以概率 \\(p\\) 让某些神经元不工作，并将其他神经元输出扩大 \\(\frac{1}{1-p}\\) 倍

- Dropout 使神经元随机失效，可以减少神经元之间的耦合，一定程度上简化模型

- 每次 Dropout 得到的模型都不相同，不同模型之间的过拟合可能相互抵消，从而整体上减少过拟合

## LRN

- Local Response Normalization，即局部响应归一化

### 侧抑制

- 在神经生物学中，被激活的神经元会抑制相邻神经元

- LRN 通过在同一层的不同特征通道之间进行归一化，实现侧抑制

### 计算过程

- \\(N\\) 是当前层总的特征通道数

- \\(a\_{x,y}^{i}\\) 表示第 \\(i\\) 个通道 \\((x,y)\\) 位置的激活输出，\\(b\_{x,y}^{i}\\) 表示 \\(a\_{x,y}^{i}\\) 的 LRN 输出

- \\(k=2, \ n=5, \ \alpha=1e-4, \ \beta=0.75\\) 均为超参数

- LRN 结果如下：

	$$ b\_{x,y}^{i} = a\_{x,y}^{i} \ / \ \left( k + \alpha \sum\_{j=\max{(0, \ i-n/2})}^{\min{(N-1, \ i+n/2)}} \left( a\_{x,y}^{j} \right)^{2} \right) ^{\beta} $$

	- 因此 \\(b\_{x,y}^{i}\\) 是以 \\(i\\) 通道为中心的 \\(n\\) 个相邻通道上， \\((x,y)\\) 位置激活值的某种组合

	- \\(a\_{x,y}^{i}\\) 越大，相邻通道 \\((x,y)\\) 位置上的输出被抑制得越严重

## Batch Normalization

### Covariate Shift

- 浅层网络参数的改变导致深层网络的输入分布发生变化，是神经网络训练困难的原因

### Whitening

- 对输入进行白化（零均值、单位标准差）能够加速网络收敛

### Batch Normalization

- 核心思想是消除浅层权重变化对深层输入分布的影响

- 训练过程如下：

	- 计算输入 \\(x\\) 的均值 \\(\mu\_{B}\\) 与方差 \\(\sigma\_{B}^{2}\\)：

		$$ \mu\_{B} = \frac{1}{m} \sum\_{i=1}^{m} x\_{i} \qquad \sigma\_{B}^{2} = \frac{1}{m} \sum\_{i=1}^{m} (x\_{i} - \mu\_{B})^{2} $$

	- 将输入 \\(x\\) 归一化为零均值、单位方差的 \\(\hat{x\_{i}}\\)：

		$$ \hat{x\_{i}} = \frac{x\_{i} - \mu\_{B}}{\sqrt{\sigma\_{B}^{2} + \epsilon}} $$

	- 由于 \\(\hat{x\_{i}}\\) 改变了输入的分布，通过两个可学习的参数还原原始分布：

		$$ y\_{i} = \gamma \hat{x\_{i}} + \beta $$

		- \\(y\_{i}\\) 即为 Batch Normalization 后的输出

		- 训练稳定时，\\(\gamma\\) 近似等于 \\(x\_{i}\\) 的标准差 \\(\sigma\_{B}\\)，\\(\beta\\) 近似等于 \\(x\_{i}\\) 的均值 \\(\mu\_{B}\\)

		- 由于 \\(\hat{x\_{i}}\\) 分布固定， \\(y\_{i}\\) 的分布不再受浅层网络参数的影响，可以加速训练过程

- 测试过程如下：

	- Batch Normalization 是为了方便训练，测试时不应该使用；但由于训练时作为网络的整体，直接拿掉会影响结果

	- 首先计算训练过程中所有的 batch 的均值和方差：

		$$ \mu = E \left\[\mu\_{B}\right\] \qquad \sigma^{2} = \frac{m}{m-1} E [\sigma\_{B}^{2}] $$

	- 此时 Batch Normalization 的输出值为：

		$$ y = \gamma \left( \frac{x-\mu}{\sqrt{\sigma^{2}+\epsilon}} \right) + \beta = \frac{\gamma}{\sqrt{\sigma^{2}+\epsilon}} \cdot x + \left( \beta - \frac{\gamma \mu}{\sqrt{\sigma^{2}+\epsilon}}\right) $$

		- 由于 \\(\gamma\\) 是 \\(\sigma\\) 的近似，\\(\beta\\) 是 \\(\mu\\) 的近似，上式近似等于：

			$$ y = x $$
			
			- 相当于不使用 Batch Normalization

- 在 CNN 中，通常为每个特征图设置一个 \\(\gamma, \ \beta\\)，以减少参数量