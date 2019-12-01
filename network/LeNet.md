# $\mathrm{LeNet}$

## 网络结构

- 从输入到输出一共 $8$ 层

  <center>
  <img src="images/lenet.png"/>
  </center>

### 输入层 $\mathrm{Input}$

- 输入为 $32 \times 32$ 的灰度图

### 卷积层 $\mathrm{Conv1}$

- 对 $\mathrm{Input}$ 数据进行卷积，特征图维度为 $28 \times 28$

  - 卷积核大小为 $5$，步长为 $1$，不进行 $\mathrm{padding}$

### $\mathrm{Pooling}$ 层 $\mathrm{Pool1}$

- 对 $\mathrm{Conv1}$ 结果进行 $\mathrm{Pooling}$，特征图维度为 $14 \times 14$

  - 步长为 $2$ 的 $2 \times 2 \ \mathrm{max \ pooling}$

### 卷积层 $\mathrm{Conv2}$

- 对 $\mathrm{Pool1}$ 结果进行卷积，特征图维度为 $10 \times 10$

  - 卷积核大小为 $5$，步长为 $1$，不进行 $\mathrm{padding}$

### $\mathrm{Pooling}$ 层 $\mathrm{Pool2}$

- 对 $\mathrm{Conv2}$ 结果进行 $\mathrm{Pooling}$，特征图维度为 $5 \times 5$

  - 步长为 $2$ 的 $2 \times 2 \ \mathrm{max \ pooling}$

### 全连接层 $\mathrm{FC1}$

- 与 $\mathrm{Pool2}$ 进行全连接，生成向量维度为 $120$

### 全连接层 $\mathrm{FC2}$

- 与 $\mathrm{FC1}$ 进行全连接，生成向量维度为 $84$

### 输出层 $\mathrm{Output}$

- 与 $\mathrm{FC2}$ 进行全连接，生成向量维度为 $10$

  - 生成向量经过 $\mathrm{softmax}$ 处理，分别代表每一类的概率