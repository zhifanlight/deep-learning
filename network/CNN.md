<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Convolutional Neural Networks

## 背景介绍

- 与 BP 网络相比，CNN 的隐藏层细分为卷积层、Pooling 层、全连接层

- 输入是二维图像而非一维向量，因此可以学到像素点的空间关系

- 卷积层、Pooling 层的梯度计算方式与 BP 的全连接层不同，但训练过程相同

![img](images/lenet.png)

### 卷积层

- 输入图像与不同的卷积核做卷积，得到不同的特征图像

- 在训练时，需要进行卷积核、偏置项等参数的学习

- 输入尺寸 \\(i\\)、输出尺寸 \\(o\\)、padding 长度 \\(p\\)、卷积核大小 \\(k\\)、步长 \\(s\\) 的关系为：

	$$ o = \left \lfloor \frac{i-k+2p}{s} \right \rfloor + 1 $$

- \\(1 \times 1\\) 卷积核的作用：

	- 整合不同通道的信息

	- 实现通道层面的降维，以减少计算

### Pooling 层

- 降低特征图像的维度，分为 max-pooling 和 average-pooling

	- max-pooling

		- 取邻域最大值作为当前像素点的像素值
	
	- average-pooling

		- 取邻域平均值作为当前像素点的像素值

- 卷积层生成的特征图通常包含冗余信息，需要通过 Pooling 来去除冗余

	- 如果使用 max-pooling，相当于只考虑最大激活值

	- 而使用 average-pooling，相当于考虑了所有冗余信息

- 通常使用 max-pooling；但分类时的最后一个池化层可以进行 global average pooling

- 没有参数，在训练时无需学习

- 输入尺寸 \\(i\\)、输出尺寸 \\(o\\)、池化核大小 \\(k\\)、步长 \\(s\\) 的关系为：

	$$ o = \left \lceil \frac{i-k}{s} \right \rceil + 1 $$

## 数学推导

### 单通道输入、单通道特征图

#### 卷积层

- 定义如下：

	- \\(w\_{m, \ n}^{l} \ \ \\)：第 \\(l\\) 层的卷积核权值

	- \\(b^{l} \ \ \\)：第 \\(l\\) 层的偏置

	- \\(E \ \ \\)：输出层误差

	- \\(x\_{i, \ j}^{l} \ \ \\)：第 \\(l\\) 层 \\((i, \ j)\\) 像素点的加权输入
	
		$$ x\_{i, \ j}^{l} = \sum\_{m}\sum\_{n} w\_{m, \ n}^{l} \cdot z\_{i+m, \ j+n}^{l-1} + b^{l} $$

	- \\(z\_{i, \ j}^{l} \ \ \\)：第 \\(l\\) 层 \\((i, \ j)\\) 像素点的预测输出
	
		$$ z\_{i,\ j}^{l} = f(x\_{i, \ j}^{l}) $$
	
	- \\(\delta\_{i, \ j}^{l} \ \ \\)：第 \\(l\\) 层 \\((i, \ j)\\) 像素点的误差项
	
		$$ \delta\_{i, \ j}^{l} = \frac{\partial{E}}{\partial{x\_{i, \ j}^{l}}} $$

- 由梯度下降可得：

	$$
	\\left\\{ \begin{matrix}
	w\_{m, \ n}^{l} \leftarrow w\_{m, \ n}^{l} - \eta \cdot \frac {\partial{E}}{\partial{w\_{m, \ n}^{l}}} \\\\
	b^{l} \leftarrow b^{l} - \eta \cdot \frac {\partial{E}}{\partial{b^{l}}}
	\end{matrix} \\right\.
	$$

- 对权重项应用链式法则：

	$$ \frac{\partial{E}}{\partial{w\_{m, \ n}^{l}}} = \sum\_{i} \sum\_{j}\frac{\partial{E}}{\partial{x\_{i, \ j}^{l}}} \cdot \frac{\partial{x\_{i, j}^{l}}}{\partial{w\_{m, \ n}^{l}}} = \sum\_{i} \sum\_{j} \delta\_{i, \ j}^{l} \cdot z\_{i+m, \ j+n}^{l-1} $$

	- 于是 \\(w\_{ji}\\) 更新公式为：

		$$ w\_{m, \ n}^{l} \leftarrow w\_{m, \ n}^{l} - \eta \cdot \sum\_{i} \sum\_{j} \delta\_{i, \ j}^{l} \cdot z\_{i+m, \ j+n}^{l-1} $$

- 对偏置项应用链式法则：

	$$ \frac{\partial{E}}{\partial{b^{l}}} = \sum\_{i} \sum\_{j}\frac{\partial{E}}{\partial{x\_{i, \ j}^{l}}} \cdot \frac{\partial{x\_{i, j}^{l}}}{\partial{b^{l}}} = \sum\_{i} \sum\_{j} \delta\_{i, \ j}^{l} $$
	
	- 于是：

		$$ b^{l} \leftarrow b^{l} - \eta \cdot \sum\_{i} \sum\_{j} \delta\_{i, \ j}^{l} $$

- 误差项 \\( \delta\_{i, \ j}^{l} = \frac{\partial{E}}{\partial{x\_{i, \ j}^{l}}} \\) 计算如下：

	- 假设受输入 \\(x\_{i, \ j}^{l}\\) 影响的区域为 \\(Q\\)，则：

		$$ \frac{\partial{E}}{\partial{x\_{i, \ j}^{l}}} = \sum\_{Q} \frac{\partial{E}}{\partial{x\_{Q}^{l+1}}} \cdot \frac{\partial{x\_{Q}^{l+1}}}{\partial{x\_{i, \ j}^{l}}} $$

	- 假设 \\(X = \left[ \begin{matrix} a & b & c \\\\ d & e & f \\\\ g & h & j \end{matrix} \right]\\)，\\(W = \left[ \begin{matrix} w & x \\\\ y & z \end{matrix} \right]\\)，\\(Y = \left[ \begin{matrix} p & q \\\\ r & s \end{matrix} \right]\\)，\\(\delta\_{X} = \left[ \begin{matrix} \delta\_{a} & \delta\_{b} & \delta\_{c} \\\\ \delta\_{d} & \delta\_{e} & \delta\_{f} \\\\ \delta\_{g} & \delta\_{h} & \delta\_{j} \end{matrix} \right]\\)，\\(\delta\_{Y} = \left[ \begin{matrix} 0 & 0 & 0 & 0 \\\\ 0 & \delta\_{p} & \delta\_{q} & 0 \\\\ 0 & \delta\_{r} & \delta\_{s} & 0 \\\\ 0 & 0 & 0 & 0 \end{matrix} \right]\\)

- 正向计算时：

	$$ Y = X \otimes W $$

	- 由卷积定义：

		$$ \left\\{ \begin{matrix} p = aw + bx + dy + ez \\\\ q = bw + cx + ey + fz \\\\ r = dw + ex + gy + hz \\\\ s = ew + fx + hy + jz \end{matrix} \right. $$

- 反向传播时：

	$$ \delta\_{X} = \delta\_{Y} \otimes Rot(W) $$

	- 其中 \\(Rot(W) = \left[ \begin{matrix} z & y \\\\ x & w \end{matrix} \right]\\) 表示对卷积核 \\(W\\) 同时进行水平翻转和竖直翻转

	- 以 \\(\delta\_{a}, \ \delta\_{e}\\) 为例，偏导数计算如下：

		$$ \frac{\partial{E}}{\partial{a}} = \frac{\partial{E}}{\partial{p}} \cdot \frac{\partial{p}}{\partial{a}} = \delta\_{p} \cdot w $$ 
				
		$$ \frac{\partial{E}}{\partial{e}} = \frac{\partial{E}}{\partial{p}} \frac{\partial{p}}{\partial{e}} + \frac{\partial{E}}{\partial{q}} \frac{\partial{q}}{\partial{e}} + \frac{\partial{E}}{\partial{r}} \frac{\partial{r}}{\partial{e}} + \frac{\partial{E}}{\partial{s}} \frac{\partial{s}}{\partial{e}} = \delta\_{p} \cdot z + \delta\_{q} \cdot y + \delta\_{r} \cdot x + \delta\_{s} \cdot w $$

- 在实际计算时，由于进行了 \\(im2col\\) 和 \\(col2im\\)，不用翻转卷积核；但需要对 \\(im2col\\) 后的卷积核进行转置

	- 正向计算：

		- 进行 \\(im2col\\)：

			$$ W = \left[ \begin{matrix} w & x & y & z \end{matrix} \right], \quad X = \left[ \begin{matrix} a & b & d & e \\\\ b & c & e & f \\\\ d & e & g & h \\\\ e & f & h & j \end{matrix} \right] $$

		- 进行矩阵乘法：

			$$ Y = WX = \left[ \begin{matrix} wa + xb + yd + ze \\\\ wb + xc + ye + zf \\\\ wd + xe + yg + zh \\\\ we + xf + yh + zj \end{matrix} \right]^{T} $$

	- 反向传播：

		- 误差矩阵不使用 padding：

			$$ \delta\_{Y} = \left[ \begin{matrix} \delta\_{p} & \delta\_{q} \\\\ \delta\_{r} & \delta\_{s} \end{matrix} \right] $$

		- 进行 \\(im2col\\)：

			$$ W^{T} = \left[ \begin{matrix} w \\\\ x \\\\ y \\\\ z \end{matrix} \right], \quad \delta\_{Y} = \left[ \begin{matrix} \delta\_{p} & \delta\_{q} & \delta\_{r} & \delta\_{s} \end{matrix} \right] $$

		- 进行矩阵乘法：

			$$ \delta\_{X} = W^{T} \delta\_{Y} = \left[ \begin{matrix} w \delta\_{p} & w \delta\_{q} & w \delta\_{r} & w \delta\_{s} \\\\ x \delta\_{p} & x \delta\_{q} & x \delta\_{r} & x \delta\_{s} \\\\ y \delta\_{p} & y \delta\_{q} & y \delta\_{r} & y \delta\_{s} \\\\ z \delta\_{p} & z \delta\_{q} & z \delta\_{r} & z \delta\_{s} \end{matrix} \right] $$

		- 进行 \\(col2im\\)：

			$$ \delta\_{X} = \left[ \begin{matrix} w \delta\_{p} & w \delta\_{q} + x \delta\_{p} & x \delta\_{q} \\\\ w \delta\_{r} + y \delta\_{p} & w \delta\_{s} + x \delta\_{r} + y \delta\_{q} + z \delta\_{p} & x \delta\_{s} + z \delta\_{q} \\\\ y \delta\_{r} & y \delta\_{s} + z \delta\_{r} & z \delta\_{s} \end{matrix} \right] $$

#### Pooling 层

- max-pooling

	- 根据前向计算时记录的最大值位置，将误差原封不动地传到前一层邻域最大值处

- average-pooling

	- 将后一层误差均匀地传到前一层邻域内的所有像素点