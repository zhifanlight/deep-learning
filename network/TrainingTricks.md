<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 训练技巧

## 数据增强

- 参考 [DataPreprocess.md](../basic/DataPreprocess.md)

## 权重初始化

### 输入为固定高斯分布 

- 对于输入 \\(x\\)，假设 \\(E(x) = 0, \ Var(x) = 1\\)

- 假设输入为 \\(n\\) 维，输出为 \\(1\\) 维，激活函数为 Sigmoid，权重 \\(w \sim N(0,1) \\)：

	$$ y = \sum\_{i=1}^{n} w\_{i} x\_{i} \qquad z = \frac{1}{1+e^{-y}} $$

- 由期望、方差的线性运算可知：

	$$ E(y) = \sum\_{i=1}^{n} E(x\_{i}) \cdot E(w\_{i}) = 0 $$

	$$ Var(y) = \sum\_{i=1}^{n} Var(x\_{i}) \cdot Var(w\_{i}) = n $$

	- 关于期望、方差线性运算，参考 [Theorems.md](../basic/Theorems.md)

- 线性加权 \\(y\\) 的方差与输入维度有关，很容易落入 Sigmoid 饱和区，不利于反向传播

### 输出为固定高斯分布

- 对于输入 \\(x\\)，假设 \\(E(x) = 0, \ Var(x) = 1\\)

- 假设输入为 \\(n\\) 维，输出为 \\(1\\) 维，激活函数为 Sigmoid，权重 \\(w \sim N \left( 0,\sqrt{\frac{1}{n}} \right) \\)：

	$$ y = \sum\_{i=1}^{n} w\_{i} x\_{i} \qquad z = \frac{1}{1+e^{-y}} $$

- 由期望、方差的线性运算可知：

	$$ E(y) = \sum\_{i=1}^{n} E(x\_{i}) \cdot E(w\_{i}) = 0 $$

	$$ Var(y) = \sum\_{i=1}^{n} Var(x\_{i}) \cdot Var(w\_{i}) = 1 $$

- 线性加权 \\(y\\) 的方差与输入维度无关，落入 Sigmoid 饱和区概率较低，方便反向传播

### 输出为固定均匀分布

- 对于输入 \\(x\\)，假设 \\(E(x) = 0, \ Var(x) = 1\\)

- 假设输入为 \\(n\\) 维，输出为 \\(1\\) 维，激活函数为 Sigmoid，权重 \\(w \sim U \left( -\sqrt{\frac{1}{n}}, \sqrt{\frac{1}{n}} \right) \\)

- 均匀分布的期望、方差计算如下：

	$$ E(x) = \frac{a+b}{2} \qquad Var(x) = \frac{(b-a)^{2}}{12} $$

- 由期望、方差的线性运算可知：

	$$ E(y) = \sum\_{i=1}^{n} E(x\_{i}) \cdot E(w\_{i}) = 0 $$

	$$ Var(y) = \sum\_{i=1}^{n} Var(x\_{i}) \cdot Var(w\_{i}) = \frac{1}{3} $$

- 线性加权 \\(y\\) 的方差与输入维度无关，落入 Sigmoid 饱和区概率较低，方便反向传播

### Xavier

- 由于 ReLU 无法进行数据压缩，输入 \\(x\\) 方差不一定为 \\(1\\)

- 保证输入和输出的方差尽量相等，使网络信息更好的流动

- 假设输入为 \\(n\\) 维，输出为 \\(m\\) 维，激活函数为 ReLU 或 Tanh，权重 \\(w \sim U(-\sqrt{\frac{6}{m+n}}, \sqrt{\frac{6}{m+n}})\\)

- 由期望、方差的线性运算可知：

	$$ E(y\_{j}) = \sum\_{i=1}^{n} E(x\_{i}) \cdot E(w\_{i}) = 0 $$
	
	$$ Var(y\_{j}) = \sum\_{i=1}^{n} Var(x\_{i}) \cdot Var(w\_{i}) = \frac{2n}{m+n} \cdot Var(x) $$

	- 当输入、输出维度近似相等时，\\(Var(y) = Var(x)\\)

- 激活函数是 Tanh 时，深层权重可以保持良好分布

- 激活函数是 ReLU 时，深层权重迅速向 \\(0\\) 靠拢

### He Initialization

- 在 ReLU 网络中，假定每一层只有一半的神经元被激活

- 如果输入 \\(x\\) 满足 \\(E(x) = 0, \ Var(x) = 1\\)，权重初始化为 \\(w \sim U(-\sqrt{\frac{2}{n}}, \sqrt{\frac{2}{n}})\\)

	- 由期望、方差的线性运算可知：

		$$ E(y) = \frac{1}{2} \sum\_{i=1}^{n} E(x\_{i}) \cdot E(w\_{i}) = 0 $$

		$$ Var(y) = \frac{1}{2} \sum\_{i=1}^{n} Var(x\_{i}) \cdot Var(w\_{i}) = \frac{1}{3} $$

- 如果输入 \\(x\\) 不满足上述约束，权重初始化为 \\(w \sim U(-\sqrt{\frac{12}{m+n}}, \sqrt{\frac{12}{m+n}})\\)

	- 由期望、方差的线性运算可知：

		$$ E(y\_{j}) = \frac{1}{2} \sum\_{i=1}^{n} E(x\_{i}) \cdot E(w\_{i}) = 0 $$

		$$ Var(y\_{j}) = \frac{1}{2} \sum\_{i=1}^{n} Var(x\_{i}) \cdot Var(w\_{i}) = \frac{2n}{m+n} \cdot Var(x) $$

## 正则化

- 通过 \\(L\_{1}\\) 或 \\(L\_{2}\\) 正则项，使参数大部分为 0 或尽量接近 0，一定程度上简化模型

- 关于正则化，参考 [Regularization.md](../basic/Regularization.md)

## Dropout

- Caffe 实现：

	- 训练时以概率 \\(p\\) 让某些神经元不工作，并将其他神经元输出扩大 \\(\frac{1}{1-p}\\) 倍

	- 测试时不进行 Dropout

	- 需要设置 drop_ratio，即神经元不工作的概率

- Tensorflow 实现：

	- 训练时以概率 \\(q\\) 让某些神经元工作，并将其输出扩大为 \\(\frac{1}{q}\\) 倍

	- 测试时不进行 Dropout

	- 需要设置 keep_prob，即神经元工作的概率

- 使用 Dropout 层对预训练的网络进行 fine tuning 时，所有参数都要乘以 \\(\frac{1}{p}\\)

- 从正则化角度，Dropout 强迫一个神经元和随机挑选出来的神经元共同工作，可以减少神经元之间的依赖，增强泛化能力

- 从 Bagging 角度，每次 Dropout 得到的模型都不相同，不同模型之间的过拟合可能相互抵消，从而整体上减少过拟合

- 在使用 Dropout 后，反向传播只针对网络的一部分，参数更新比较随机，训练时间较长

## LRN

- Local Response Normalization，即局部响应归一化

### 侧抑制

- 在神经生物学中，被激活的神经元会抑制相邻神经元

- LRN 通过在同一层的不同特征通道之间进行归一化，实现侧抑制

### 计算过程

- \\(N\\) 是当前层总的特征通道数

- \\(a\_{x,y}^{\ i}\\) 表示第 \\(i\\) 个通道 \\((x,y)\\) 位置的激活输出，\\(b\_{x,y}^{\ i}\\) 表示 \\(a\_{x,y}^{\ i}\\) 的 LRN 输出

- \\(k=2, \ n=5, \ \alpha=1e-4, \ \beta=0.75\\) 均为超参数

- LRN 结果如下：

	$$ b\_{x,y}^{\ i} = a\_{x,y}^{\ i} \ / \ \left( k + \alpha \sum\_{j=\max{(0, \ i-n/2})}^{\min{(N-1, \ i+n/2)}} \left( a\_{x,y}^{\ j} \right)^{2} \right) ^{\beta} $$

	- 因此 \\(b\_{x,y}^{\ i}\\) 是以 \\(i\\) 通道为中心的 \\(n\\) 个相邻通道上， \\((x,y)\\) 位置激活值的某种组合

	- \\(a\_{x,y}^{\ i}\\) 越大，相邻通道 \\((x,y)\\) 位置上的输出被抑制得越严重

### 工程实现

- 在 Caffe 中有两种归一化方式：通道内、通道间

- LRN 层通常放在 ReLU 层之后

## Batch Normalization

- 通过固定每一层的输入，加速网络收敛

- 关于 Batch Normalization，参考 [BatchNormalization.md](BatchNormalization.md)