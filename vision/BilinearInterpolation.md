# 双线性插值

## 背景介绍

- 通过在 $x, \ y$ 两个方向分别插值，实现对图像的平滑缩放

## 原理分析

### 线性插值

- 已知直线上两定点 $\left( x_{1}, \ y_{1} \right), \ \left( x_{2}, \ y_{2} \right)$，两点连线间任意一点的坐标为 $\left( x, \ y \right)$

- 根据共线可得：

  $$
  \frac{y - y_{1}}{x - x_{1}} = \frac{y_{2} - y_{1}}{x_{2} - x_{1}} \quad \Rightarrow \quad y = \frac{x - x_{1}}{x_{2} - x_{1}} \cdot y_{2} + \frac{x_{2} - x}{x_{2} - x_{1}} \cdot y_{1} 
  $$

- 因此，越靠近 $\left( x_{1}, \ y_{1} \right)$，$y$ 值越接近 $y_{1}$，越靠近 $\left( x_{2}, \ y_{2} \right)$，$y$ 值越接近 $y_{2}$

### 双线性插值

- $f \left( x, \ y \right)$ 表示像素值；$\left( x, \ y \right)$ 是缩放后的图像坐标，$\left( x_{0}, \ y_{0} \right)$ 是原图像的对应坐标，$\left( x_{0}, \ y_{0} \right)$ 可通过缩放比由 $\left( x, \ y \right)$ 计算得到，缩放比定义为缩放前、后的图像尺寸之比

#### 坐标缩放

- 由 $\left( x, \ y \right)$ 计算 $\left( x_{0}, \ y_{0} \right)$ 有两种方式：

  - 角对齐：

    $$
    \left\{ \begin{matrix} x_{0} = x \cdot \mathrm{scaleX} \\ y_{0} = y \cdot \mathrm{scaleY} \end{matrix} \right.
    $$

  - 中心对齐：

    $$
    \left\{ \begin{matrix} x_{0} = \left( x + 0.5 \right) \cdot \mathrm{scaleX} - 0.5 \\ y_{0} = \left( y + 0.5 \right) \cdot \mathrm{scaleY} - 0.5 \end{matrix} \right. 
    $$

<center>
<img src="images/bilinear_interpolation_scale.png"/>
</center>

- 上图表示将 $5 \times 5$ 的图像缩放到 $3 \times 3$ 时，两种情况下分别计算的 $\left( x_{0}, \ y_{0} \right)$ 分布：

  - 角对齐时，右下方的像素值不能被充分利用

  - 中心对齐时，两张图像的几何中心重合，能充分利用边界处的像素值

- 中心对齐可由角对齐平移 $\Delta$ 得到，推导如下：

  - 假设要将 $M \times M$ 的图像缩放到 $N \times N$，因此缩放比 $\mathrm{scale} = \frac{M}{N}$

  - 由上图（中心对齐）可知，应该有 $N - 1$ 个大间隔，以及上、下、左、右各一个 $\Delta$，因此：

    $$
    \frac{M}{N} \left( N - 1 \right) + 2 \Delta = M - 1 \quad \Rightarrow \quad \Delta = 0.5 \cdot \left( \mathrm{scale} - 1 \right)
    $$

  - 而在角对齐中，$v_{0} = v \cdot \mathrm{scale}$，将 $\Delta$ 代入可得：

    $$
    v_{0} = v \cdot \mathrm{scale} + \Delta \quad \Rightarrow \quad \left\{ \begin{matrix} x_{0} = \left( x + 0.5 \right) \cdot \mathrm{scaleX} - 0.5 \\ y_{0} = \left( y + 0.5 \right) \cdot \mathrm{scaleY} - 0.5 \end{matrix} \right.
    $$

#### 插值计算

- $\left( x_{0}, \ y_{0} \right)$ 一般不为整数，因此需要用插值的方来法计算该点的像素值 $f \left( x_{0}, \ y_{0} \right)$

- 假设 $x_{0}$ 向下、向上取整分别得到 $x_{1}, \ x_{2}$， 而 $y_{0}$ 向下、向上取整分别得到 $y_{1}, \ y_{2}$

- 由线性插值公式可知：

  - $x$ 方向：

    $$
    \left\{ \begin{matrix} f \left( x_{0}, \ y_{1} \right) = \frac{x_{0} - x_{1}}{x_{2} - x_{1}} \cdot f \left( x_{2}, \ y_{1} \right) + \frac{x_{2} - x_{0}}{x_{2} - x_{1}} \cdot f \left( x_{1}, \ y_{1} \right) \\ f \left( x_{0}, \ y_{2} \right) = \frac{x_{0} - x_{1}}{x_{2} - x_{1}} \cdot f \left( x_{2}, \ y_{2} \right) + \frac{x_{2} - x_{0}}{x_{2} - x_{1}} \cdot f \left( x_{1}, \ y_{2} \right) \end{matrix} \right.
    $$

  - $y$ 方向：

    $$
    f \left( x_{0}, \ y_{0} \right) = \frac{y_{0} - y_{1}}{y_{2} - y_{1}} \cdot f \left( x_{0}, \ y_{2} \right) + \frac{y_{2} - y_{0}}{y_{2} - y_{1}} \cdot f \left( x_{0}, \ y_{1} \right)
    $$

<center>
<img src="images/bilinear_interpolation.png"/>
</center>

- 设 $S_{1}, \ S_{2}, \ S_{3}, \ S_{4}$ 分别表示四个区域的面积，由于 $x_{2} - x_{1} = y_{2} - y_{1} = 1$，综合上式可得：

  $$
  f \left( x_{0}, \ y_{0} \right) = S_{1} \cdot f \left( x_{1}, \ y_{1} \right) + S_{2} \cdot f \left( x_{2}, \ y_{1} \right) + S_{3} \cdot f \left( x_{2}, \ y_{2} \right) + S_{4} \cdot f \left( x_{1}, \ y_{2} \right)
  $$

- $\left( x_{0}, \ y_{0} \right)$ 越接近 $\left( x_{2}, \ y_{2} \right)$，$S_{3}$ 就越大，$f \left( x_{0}, \ y_{0} \right)$ 越接近 $f \left( x_{2}, \ y_{2} \right)$；其他方向同理

## $\mathrm{Python}$ 实现

```python
response = cv2.resize(image, (width, height))
```