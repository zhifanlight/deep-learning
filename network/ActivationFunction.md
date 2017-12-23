<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 激活函数

## 背景介绍

- 在神经网络中，当不使用激活函数或使用线性激活函数时，每一层输出都是上一层输入的线性函数；无论神经网络有多少层，输出都是输入的线性组合，难以拟合复杂的函数

### 激活函数特性

#### 非线性

- 对于线性激活函数，输出仅仅是输入的线性组合

#### 可微性

- 当优化方法基于梯度时，该性质必须满足

#### 单调性

- 保证单层网络是凸函数，可以优化求解

#### 输出值范围

- 特征的表示受有限权值的影响，基于梯度的优化方法更稳定

## 常用激活函数

### Softmax

- 用于多分类神经网络输出，计算如下：

	$$ f(x\_{i}) = \frac{e^{x\_{i}}}{\sum\_{j}e^{x\_{j}}} $$

#### softmax 溢出

- 在计算 softmax 时存在以下问题：

	- 某个 \\(x\_{i}\\) 太大，计算 \\(e^{x\_{i}}\\) 容易上溢

	- 所有 \\(x\_{j}\\) 都太小，计算 \\(\sum\_{j}e^{x\_{j}}\\) 容易下溢，分母为 0

- 通常先减去 \\(x\\) 中的最大值 \\(M\\)，再进行计算：

	$$ \frac{e^{x\_{i}}}{\sum\_{j}e^{x\_{j}}} = \frac{e^{-M} \cdot e^{x\_{i}}}{e^{-M} \cdot \sum\_{j}e^{x\_{j}}} = \frac{e^{x\_{i}-M}}{\sum\_{j}e^{x\_{j}-M}} $$

	- 都减去最大值 \\(M\\)，保证了不会上溢

	- 分母中至少有一项为 1，虽无法避免下溢，却能保证分母有意义

#### log softmax 溢出

- 由于上述方法无法避免下溢，当需要计算 log softmax 时存在以下问题：

	- 某个 \\(x\_{i}\\) 太小，其 softmax 值为 0，而 \\(log(0) =- \infty\\)

- 通常不计算 softmax，直接 log 值：

	$$ log \left( \frac{e^{x\_{i}}}{\sum\_{j}e^{x\_{j}}} \right) = log \left( \frac{e^{x\_{i}-M}}{\sum\_{j}e^{x\_{j}-M}} \right) = (x\_{i} - M) - log \left( \sum\_{j}e^{x\_{j}-M} \right) $$

	- 都减去最大值 \\(M\\)，保证了不会上溢

	- log 中至少有一项为 1，虽无法避免下溢，却能保证 log 有意义

### Sigmoid

- 原函数、导数分别如下：

	$$ f(x) = \frac{1}{1 + e^{-x}} \qquad f'(x) = f(x) \cdot (1 - f(x)) $$
	
	![img](images/sigmoid.png)

#### 优点

- 单调连续，将输出映射在 (0,1) 之间，输出值可作为概率，可用于输出层

#### 缺点

- 在神经网络反向传播过程中，需要用到 sigmoid 导数，但 sigmoid 导数容易饱和，导致向底层传递的梯度消失，网络参数难以有效训练

- sigmoid 输出会发生偏移现象，即均值不为 0，导致后一层神经元的输入为非 0 均值的信号

### Tanh

- 原函数、导数分别如下：

	$$ f(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}} \qquad f'(x) = 1 - f(x)^{2} $$
	
	![img](images/tanh.png)

#### 优点  

- 解决了 sigmoid 函数输出均值不为 0 的问题

- 收敛速度比 sigmoid 快

#### 缺点

- tanh 导数容易饱和，无法解决梯度消失问题，网络参数难以有效训练

### ReLU

- Rectified Linear Unit

- 原函数、导数分别如下：
	
	$$ f(x) = max \ (x, 0) \qquad f'(x) = \\left\\{ \begin{matrix} 1, \quad x \geq 0 \\\\ 0, \quad x < 0 \end{matrix} \\right\. $$
	
	![img](images/relu.png)

#### 优点

- 在 x > 0 时梯度不衰减，缓解梯度消失问题，可以有效训练深层神经网络

- 由于原函数、导数都不包含复杂的数学运算，可以加速计算

- 部分神经元输出为 0，造成了网络的稀疏性，一定程度上缓解了过拟合

#### 缺点

- 当某个神经元被 ReLU 函数抑制时，梯度难以继续更新，神经元不可逆的死亡

- 由于输出均值大于 0，偏移现象和神经元死亡共同影响神经网络收敛

- 当学习率过高时，很容易导致大部分神经元死亡

### Leaky ReLU

- 原函数、导数分别如下：

	$$ f(x) = \\left\\{ \begin{matrix} x, \quad x \geq 0 \\\\ \alpha x, \quad x < 0 \end{matrix} \\right\. \qquad f'(x) = \\left\\{ \begin{matrix} 1, \quad x \geq 0 \\\\ \alpha, \quad x < 0 \end{matrix} \\right\. $$

	![img](images/leaky_relu.png)

#### 优点

- 由于导数始终不为 0，可以解决 ReLU 神经元死亡的问题

#### 缺点

- \\(\alpha\\) 值难以凭经验确定，需要多次实验选择才能比 ReLU 更好

### PReLU

- Parametric ReLU

- 函数定义与 Leaky ReLU相同，但是将 \\(\alpha\\) 作为待学习的参数

- 根据原论文，建议将初始值设置为 0.25，不采用正则项约束

### RReLU

- Randomized ReLU

- 函数定义与 Leaky ReLU相同，但是 \\(\alpha\\) 不再固定，而是某个区间内的随机值

- 由于 \\(\alpha\\) 取值随机，在一定程度上可以起到正则项的效果

### ELU

- Exponential Linear Unit

- 原函数、导数分别如下：

	$$ f(x) = \\left\\{ \begin{matrix} x, \qquad \qquad x \geq 0 \\\\ \alpha(e^{x} - 1), \quad x < 0 \end{matrix} \\right\. \qquad f'(x) = \\left\\{ \begin{matrix} 1, \qquad x \geq 0 \\\\ \alpha e^{x}, \quad x < 0 \end{matrix} \\right\. $$

	![img](images/elu.png)

#### 优点

- 减少了两部分梯度间的差距，可以加速神经网络收敛

### Softplus

- 原函数、导数分别如下：

	$$f(x) = log \ (1 + e^{x}) \qquad f'(x) = \frac{1}{1 + e^{-x}}$$
	
	![img](images/softplus.png)

#### 优点

- softplus 导数处处连续、无处不在，可以防止神经元死亡

#### 缺点

- 由于不对称，不以 0 为中心，可能会影响神经网络的收敛

- 由于导数为 sigmoid 函数，处处小于 0，无法避免梯度消失问题

### Maxout

- 同时训练多组参数，将最大激活值作为最终的输出

- （二元）原函数、导数分别如下：

	$$ f(x) = max \ (w\_{1}^{T}x + b\_{1}, w\_{2}^{T}x + b\_{2}) \qquad f'(x) = \\left\\{ \begin{matrix} w\_{1}, \quad w\_{1}^{T}x + b\_{1} \geq w\_{2}^{T}x + b\_{2} \\\\ w\_{2} , \quad w\_{1}^{T}x + b\_{1} < w\_{2}^{T}x + b\_{2} \end{matrix} \\right\. $$

#### 优点

- 能够近似任意连续函数

- 可以像 ReLU 一样避免梯度消失，又不会像 ReLU 那样导致神经元死亡

#### 缺点

- 参数翻倍，甚至增加几倍，增加了计算量，导致网络效率降低