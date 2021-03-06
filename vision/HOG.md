# $\mathrm{HOG}$ 特征

## 基本思想

- 在边缘具体位置未知的情况下，边缘方向的分布也可以表示目标的外形轮廓

- 适合做计算机视觉中的人体检测

## 计算过程

### 图像预处理

- 由于颜色信息作用不大，通常先将 $\mathrm{RGB}$ 图像转换为灰度图像

- 通过 $\mathrm{Gamma}$ 变换，调节图像对比度，同时抑制噪声干扰

### 梯度计算

- 计算每个像素的梯度，包括大小和方向，以捕获轮廓信息

  - 大小：

    $$
    G \left( x, \ y \right) = \sqrt{G_{x} \left( x, \ y \right)^{2} + G_{y} \left( x, \ y \right)^{2}}
    $$

  - 方向：

    $$
    \alpha \left( x, \ y \right) = \arctan \left( \frac{G_{y} \left( x, \ y \right)}{G_{x} \left( x, \ y \right)} \right)
    $$

### $\mathrm{cell}$ 处理

- 将图像划分为若干个不重叠的 $\mathrm{cell}$，比如每个 $\mathrm{cell}$ 分辨率为 $6 \times 6$

- 采用 $9$ 个 $\mathrm{bin}$ 统计每个 $\mathrm{cell}$ 内的梯度，梯度方向决定统计区间，梯度大小作为统计权重

### $\mathrm{block}$ 处理

- 由于光照强度变化、前景-背景对比，梯度强度变化范围可能特别大，需要做归一化

- 对每个 $\mathrm{block}$ 向量分别归一化，串联所有归一化后的 $\mathrm{block}$ 向量作为最终 $\mathrm{HOG}$ 特征

- 把相邻（比如 $2 \times 2$ 个）$\mathrm{cell}$ 组成更大的 $\mathrm{block}$，将一个 $\mathrm{block}$ 内的所有 $\mathrm{cell}$ 向量串联起来组成该 $\mathrm{block}$ 的向量