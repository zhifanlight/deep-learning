# 感受野

## 背景介绍

- 特征图上的一个点所对应的输入层区域大小

## 感受野计算

- 在计算特征图维度、感受野时，池化层相当于没有 $\mathrm{padding}$ 的卷积层

- 第 $n$ 层特征图在 $\mathrm{CNN}$ 输入层上的感受野计算如下：

  - 假设第 $i$ 层步长为 $s$，卷积核大小为 $k$，第 $n$ 层特征图在第 $i + 1$ 层上的感受野为 $\mathrm{out}$，在第 $i$ 层上的感受野为 $\mathrm{in}$，递推如下：

  $$
  \mathrm{in} = \left( \mathrm{out} - 1 \right) \cdot s + k
  $$

  - 初始化：

    $$
    \mathrm{out} = 1
    $$

  - 从后向前递推，得到输入层的感受野

## 坐标映射

### 从特征图到输入图像

- 假设第 $i$ 层步长为 $s$，卷积核大小为 $k$，$\mathrm{padding}$ 值为 $p$

- 第 $i$ 层上的位置为 $\mathrm{in}$，第 $i + 1$ 层上的位置为 $\mathrm{out}$，递推如下：

  $$
  \mathrm{in} = \mathrm{out} \cdot s + \left( \frac{k - 1}{2} - p \right)
  $$

- 如果要计算任意两个特征图间的坐标映射关系，只需对上式进行简单嵌套