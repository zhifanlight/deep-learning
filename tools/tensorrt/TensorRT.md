# $\mathrm{TensorRT}$

## 加速原理

- 以 $\mathrm{Inception \ v1}$ 的节点为例，原始结构如下：

  - 卷积层的计算实际分为两步：计算卷积，加偏置项

  <center>
  <img src="images/origin.png"/>
  </center>

### 删除无关层

- 删除没有用到的输出层和相关层，减少计算量

### $\mathrm{Concat}$ 优化

- 直接预分配连续显存，并将计算结果写入到对应区域；不再进行 $\mathrm{Concat}$ 操作

### 低精度计算

- 使用半精度甚至四分之一精度进行计算，精度几乎没有损失，速度提升明显

### 内核融合

- 垂直方向：将多个串行节点节点合并进一个计算节点，减少 $\mathrm{GPU}$ 内核启动次数

  - 将卷积层和激活层合并得到 $\mathrm{CBR}$（$\mathrm{Convolution + Bias + ReLU}$）如下：

  <center>
  <img src="images/vertical.png"/>
  </center>

- 水平方向：合并输入相同、操作相同的层，分别计算每一部分，减少内核启动次数

  - 将来自同一输入的 $1 \times 1$ 卷积后，得到的计算图如下：

  <center>
  <img src="images/horizon.png"/>
  </center>

## 基本流程

- 将 $\mathrm{Caffe}$、$\mathrm{Tensorflow}$ 等模型转换成 $\mathrm{GIE}$ 可以运行的模型，或直接从磁盘加载

  - $\mathrm{GIE}$ 全称是 $\mathrm{GPU \ Inference \ Engine}$

  - 在转换过程中，进行上述优化

- 将数据拷贝到 $\mathrm{GPU}$ 并通过 $\mathrm{GIE}$ 计算，将计算结果拷回 $\mathrm{CPU}$

- 对于 $\mathrm{TensorRT}$ 原生不支持的层，可以自己写插件实现