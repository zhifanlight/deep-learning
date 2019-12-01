# $\mathrm{Fast \ R-CNN}$

## 思想

- 将特征提取、分类以及 $\mathrm{Bounding \ Box}$ 回归统一到一个 $\mathrm{CNN}$ 中

- 算法流程：

  - 选出一部分可能是目标物体的区域（$\mathrm{ROI}$）

  - 对 $\mathrm{CNN}$ 进行 $\mathrm{fine \ tuning}$

  - 对 $\mathrm{ROI}$ 区域进行特征提取、分类以及 $\mathrm{Bounding \ Box}$ 回归

- 对整张图像做卷积，减少重复计算，加速计算：

  - 训练过程比 $\mathrm{R-CNN}$ 快 $9$ 倍

  - 测试过程比 $\mathrm{R-CNN}$ 快 $213$ 倍

- 特征提取、分类以及 $\mathrm{Bounding \ Box}$ 回归用同一个网络实现，不需要额外存储空间

- 检测范围：背景类、$20$ 个前景类

- 由于使用了 $\mathrm{Selective \ Search}$（$\mathrm{2 - 3}$ 秒），依然无法满足实时应用

## 候选区域选择

- 采用与 $\mathrm{R-CNN}$ 相同的 $\mathrm{Selective \ Search}$，得到约 $\mathrm{2000}$ 个候选区域

## 网络结构

<center>
<img src="images/fast_rcnn.png"/>
</center>

## $\mathrm{CNN \ fine-tuning}$

### 模型结构修改

- 将最后的 $\mathrm{max \ pooling}$ 层替换为 $\mathrm{ROI \ pooling}$ 层

  - 对于最后 $\mathrm{max \ pooling}$ 层每个通道，将特征图划分为 $H \times W$ 块，在每块上进行 $\mathrm{max \ pooling}$，得到尺寸固定的输出 $C \times H \times W$

  - 后续全连接层的输入尺寸需要保持一致，而 $\mathrm{Region \ Proposal}$ 的大小可能不同，因此需要 $\mathrm{ROI \ pooling}$ 层

- 将最后的全连接层 $\mathrm{FC8}$ 替换为两个子网络：

  - $K + 1$ 类的 $\mathrm{Softmax}$ 分类器（前景、背景）

    - 输出为 $\mathrm{K + 1}$ 维向量，表示属于每个类的概率

  - 类别相关的 $\mathrm{Bounding \ Box}$ 回归器

    - 输出为 $4K$ 维向量，分别对应每一个前景类的 $t_{x}, \ t_{y}, \ t_{w}, \ t_{h}$

    - 关于 $t_{x}, \ t_{y}, \ t_{w}, \ t_{h}$，参考 [$\mathrm{R-CNN.md}$](R-CNN.md)

- $\mathrm{CNN}$ 输入包含两部分：

  - 图像：待检测图像（受限于显存容量，短边缩放到为 $\mathrm{600}$）

  - $\mathrm{ROI}$：类别 $u$、$\mathrm{Bounding \ Box}$ 目标 $v = \left( v_{x}, \ v_{y}, \ v_{w}, \ v_{h} \right)$

### $\mathrm{fine-tuning}$

- 只对 $\mathrm{VGG}$ 的全连接层 $\mathrm{fine-tuning}$ 时效果较差，因此从 $\mathrm{conv3\_1}$ 层开始进行 $\mathrm{fine-tuning}$

- 损失函数包含两部分：

  - 分类网络的 $\mathrm{Softmax}$ 交叉熵

  - $\mathrm{Bounding \ Box}$ 网络的 $\mathrm{Smooth \ L1}$ 损失：

    $$
    L_{bb} = \sum_{i \in \left\{ x, \ y, \ w, \ h \right\}} \left[ u \geq 1 \right] \cdot \mathrm{Smooth_{L_{1}}} \left( t_{i}^{u} - v_{i} \right)
    $$

    - 其中 $\mathrm{Smooth_{L_{1}}}$ 损失定义如下：

      $$
      \mathrm{Smooth_{L_{1}}} = \left\{ \begin{matrix} 0.5 x^{2} \quad \ \quad |x| \leq 1 \\ |x| - 0.5 \quad |x| > 1 \end{matrix} \right.
      $$

- 损失函数计算如下：

  $$
  L = L_{\mathrm{cls}} \left( p, \ u \right) + \lambda \cdot L_{\mathrm{bb}}
  $$

  - 超参数 $\lambda = 1$

- 为了减少卷积层的计算量，训练时首先随机选取两张图片，之后从每张图片上选取 $64$ 个 $\mathrm{ROI}$ 区域，共 $128$ 个样本组成一个 $\mathrm{batch}$

  - 每个 $\mathrm{batch}$ 包含 $32$ 个正样本，$96$ 个负样本

  - 将 $\mathrm{IoU} \geq 0.5$ 的 $\mathrm{Region \ Proposal}$ 视为 $\mathrm{Ground \ Truth}$ 对应的类

  - 将 $0.1 < \mathrm{IoU} < 0.5$ 的 $\mathrm{Region \ Proposal}$ 视为背景类

  - 按 $0.5$ 的概率对输入图像随机翻转，进行数据增强

## 特征提取

### 坐标映射

- 将输入图像的 $\mathrm{ROI}$ 区域映射到 $\mathrm{ROI \ pooling}$ 层

  $$
  \mathrm{in} = \mathrm{out} \cdot s + \left( \frac{k - 1}{2} - p \right)
  $$

  - 关于 $\mathrm{CNN}$ 不同层间坐标映射，参考 [$\mathrm{ReceptiveField.md}$](ReceptiveField.md)

- 为简化计算，在每一层都进行 $\frac{k}{2}$ 的 $\mathrm{padding}$：

  - 当 $k$ 为奇数时，上式化简为：

    $$
    \mathrm{in} = \mathrm{out} \cdot s
    $$

  - 当 $k$ 为偶数时，上式化简为：

    $$
    \mathrm{in} = \mathrm{out} \cdot s - 0.5
    $$

  - 由于坐标值 $\mathrm{in}$ 不可能是小数，上式可近似统一为：

    $$
    \mathrm{in} = \mathrm{out} \cdot s
    $$

    - 感受野中心点坐标 $\mathrm{in}$ 只与后一层有关

    - 公式按层嵌套后，与 $\mathrm{max \ pooling}$ 响应点 $\mathrm{out}$ 对应的输入层中心点为：

      $$
      \mathrm{in} = \mathrm{out} \cdot \prod_{i} s_{i} = \mathrm{out} \cdot S
      $$

- 计算输入 $\mathrm{ROI}$ 在 $\mathrm{ROI \ pooling}$ 层上的对应区域：

  - 对于特征图上的 $\left( x', \ y' \right)$，在输入图像中的对应点为 $\left( x, \ y \right) = \left( Sx', \ Sy' \right)$

  - 计算 $\mathrm{ROI}$ 左上角坐标 $\left( x, \ y \right)$ 对应的 $\mathrm{ROI \ pooling}$ 坐标：

    $$
    x' = \left \lfloor \frac{x}{S} \right \rfloor + 1 \qquad y' = \left \lfloor \frac{y}{S} \right \rfloor + 1
    $$

  - 计算 $\mathrm{ROI}$ 右上角坐标 $\left( x, \ y \right)$ 对应的 $\mathrm{ROI \ pooling}$ 坐标：

    $$
    x' = \left \lceil \frac{x}{S} \right \rceil - 1 \qquad y' = \left \lceil \frac{y}{S} \right \rceil - 1
    $$

### 全连接层提速

- 由于每个 $\mathrm{ROI}$ 都要通过全连接层进行分类和回归，全连接层占了计算时间的 $50\%$

- 全连接层的矩阵乘法计算如下：

  $$
  y = Wx
  $$

  - 假设 $y, \ x$ 分别为 $u, \ v$ 向量，则 $W$ 维度为 $u \times v$，矩阵乘法复杂度为 $uv$

- 对 $W$ 进行 $\mathrm{SVD}$，并用前 $t$ 个奇异值近似：

  $$
  W = U \Sigma V^{T} \approx U \left( :, \ 1:t \right) \cdot \Sigma \left( 1:t, \ 1:t \right) \cdot V \left( :, \ 1:t \right)^{T}
  $$

  - 关于 $\mathrm{SVD}$，参考 [$\mathrm{MatrixDecomposition.md}$](../basic/MatrixDecomposition.md)

  - 此时全连接层计算如下：

    $$
    y = U \Sigma V^{T} x = U \left( \Sigma V^{T} x \right) = Uz
    $$

    - 相当于把一个全连接层拆分为两个，中间以一个低维（ $t$ 维）数据相连

  - 由于 $\Sigma$ 为对角矩阵，此时全连接层的计算复杂度为 $t \left( v + u \right)$

    - 当 $t < \min \left( u, \ v \right)$ 时，可以实现全连接层提速

- 在 $\mathrm{Fast \ R-CNN}$ 中使用 $\mathrm{SVD}$，可以将速度提高 $30\%$，基本不会损失精度