# 卷积

#### $\mathrm{CNN}$ 中的卷积，实际是指互相关

## 互相关

- 直接进行对位乘法，记作 $I \otimes K$

  $$
  I \otimes K = \left[ \begin{matrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{matrix} \right] \otimes \left[ \begin{matrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{matrix} \right] = \left[ \begin{matrix} 1 & 4 & 9 \\ 16 & 25 & 36 \\ 49 & 64 & 81 \end{matrix} \right]
  $$

- 即：

  $$
  \left( I \otimes K \right)_{ij} = \sum_{m = 0}^{k_{1} - 1} \sum_{n = 0}^{k_{2} - 1} I \left( i + m, \ j + n \right) \cdot K \left( m, \ n \right)
  $$

## 卷积

- 先翻转卷积核，再进行对位乘法，记作 $I * K$

  $$
  I * K = \left[ \begin{matrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{matrix} \right] * \left[ \begin{matrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{matrix} \right] = \left[ \begin{matrix} 1 & 2 & 3 \\ 4 & 5 & 6 \\ 7 & 8 & 9 \end{matrix} \right] \otimes \left[ \begin{matrix} 9 & 8 & 7 \\ 6 & 5 & 4 \\ 3 & 2 & 1 \end{matrix} \right] = \left[ \begin{matrix} 9 & 16 & 21 \\ 24 & 25 & 24 \\ 21 & 16 & 9 \end{matrix} \right]
  $$

- 即：

  $$
  \begin{align*}
  \left( I * K \right)_{ij} &= \sum_{m = 0}^{k_{1} - 1} \sum_{n = 0}^{k_{2} - 1} I \left( i - m, \ j - n \right) \cdot K \left( m, \ n \right) \newline
  & = \sum_{m = 0}^{k_{1} - 1} \sum_{n = 0}^{k_{2} - 1} I \left( i, \ j \right) \cdot K \left( -m, \ -n \right)
  \end{align*}
  $$

## 卷积加速

### 卷积定理

- 参考 [$\mathrm{FourierTransform.md}$](../basic/FourierTransform.md)

- 只有当卷积核较大时，加速效果才明显

### 矩阵展开

- 假设 $X = \left[ \begin{matrix} a & b & c \\ d & e & f \\ g & h & j \end{matrix} \right]$，$W = \left[ \begin{matrix} w & x \\ y & z \end{matrix} \right]$，$Y = \left[ \begin{matrix} p & q \\ r & s \end{matrix} \right]$

- 进行 $\mathrm{im2col}$：

  $$
  W = \left[ \begin{matrix} w & x & y & z \end{matrix} \right], \quad X = \left[ \begin{matrix} a & b & d & e \\ b & c & e & f \\ d & e & g & h \\ e & f & h & j \end{matrix} \right]
  $$

  - $X$ 的每一列表示一次卷积位置

- 进行矩阵乘法：

  $$
  Y = WX = \left[ \begin{matrix} wa + xb + yd + ze \\ wb + xc + ye + zf \\ wd + xe + yg + zh \\ we + xf + yh + zj \end{matrix} \right]^{T} = \left[ \begin{matrix} p & q & r & s \end{matrix} \right]
  $$

  - $Y$ 的每一列表示一次卷积结果

## 转置卷积

- 假设 $X = \left[ \begin{matrix} a & b & c \\ d & e & f \\ g & h & j \end{matrix} \right]$，$W = \left[ \begin{matrix} w & x \\ y & z \end{matrix} \right]$，$Y = \left[ \begin{matrix} p & q \\ r & s \end{matrix} \right]$

- 转置卷积定义为 $X = W^{T} Y$

  - 转置卷积可以从输出特征图恢复输入特征图的形状

  - 转置卷积的卷积核与普通卷积的卷积核没有数值关联

- 进行 $\mathrm{im2col}$：

  $$
  W^{T} = \left[ \begin{matrix} w \\ x \\ y \\ z \end{matrix} \right], \quad Y = \left[ \begin{matrix} p & q & r & s \end{matrix} \right]
  $$

- 进行矩阵乘法：

  $$
  X = W^{T} Y = \left[ \begin{matrix} w p & w q & w r & w s \\ x p & x q & x r & x s \\ y p & y q & y r & y s \\ z p & z q & z r & z s \end{matrix} \right]
  $$

- 进行 $\mathrm{col2im}$：

  $$
  X = \left[ \begin{matrix} w p & w q + x p & x q \\ w r + y p & w s + x r + y q + z p & x s + z q \\ y r & y s + z r & z s \end{matrix} \right]
  $$

- 转置卷积的计算过程，与卷积的反向传播计算方式相同