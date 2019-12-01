# 图像平滑

## 原理分析

- 消除或减少噪声影响

- 用每个像素周围的像素值计算当前像素值

## 常用方法

### 均值滤波

- 用邻域内像素点的平均值作为当前点像素值

- 对噪音图像敏感，即使有少量噪声点存在较大差异，也会导致平均值的明显波动

### 中值滤波

- 用邻域内像素点的中值作为当前点像素值

- 滤波器为正方形，且尺寸只能为奇数

- 可以较好的消除椒盐噪声，较好的保存图像边缘信息

### 高斯滤波

- 由于图像的连续性，距离越近的点权重越大，距离越远的点权重越小

- 通过二维高斯分布计算高斯核：

  $$
  G \left( x, \ y \right) = \frac{1}{2 \pi \sigma^{2} } \exp \left( -\frac{ \left( x - \mu_{x} \right)^{2} + \left( y - \mu_{y} \right)^ {2}} {2 \sigma^{2}} \right)
  $$

- 对高斯核进行缩放，使所有 $\mathrm{cell}$ 和为 $1$，并用缩放后的高斯核对图像进行卷积

- 高斯核的尺寸只能为奇数：$ \mu_{x}, \ \mu_{y} = \frac{kx}{2}, \ \frac{ky}{2}$

### 双边滤波

- 同时考虑空间距离与像素差值的影响，可以在保留边缘信息的同时去除图片噪声

- $k$ 为归一化系数；$p \left( x, \ y \right)$ 表示 $\left( x, \ y \right)$ 点的像素值；$s \left( a, \ b, \ x, \ y \right)$ 表示 $\left( a, \ b \right)$ 和 $\left( x, \ y \right)$ 两个点空间距离的高斯值；$c \left( x, \ y \right)$ 表示 $x, \ y$ 两种像素值距离的高斯值

  $$
  \left\{ \begin{matrix} c \left( x, \ y \right) = \exp \left( -\frac{ \left( x - y \right)^{2}} {2 \sigma_{c}^{2}} \right) \\ s \left( a, \ b, \ x, \ y \right) = \exp \left( -\frac{ \left( a - x \right)^{2} + \left( b - y \right)^{2}} {2 \sigma_{s}^{2}} \right) \\ k \left( x, \ y \right) = \iint s \left( a, \ b, \ x, \ y \right) \cdot c \left( p \left( a, \ b \right), \ p \left( x, \ y \right) \right) \mathrm{d} a \mathrm{d} b \\ G \left( x, \ y \right) = k \left( x, \ y \right)^{-1} \iint p \left( a, \ b \right) \cdot s \left( a, \ b, \ x, \ y \right) \cdot c \left( p \left( a, \ b \right), \ p \left( x, \ y \right) \right) \mathrm{d} a \mathrm{d} b \end{matrix} \right.
  $$

## $\mathrm{Python}$ 实现

### 均值滤波

```python
response = cv2.blur(image, (kx, ky))
```

### 中值滤波

```python
borderType = cv2.BORDER_REPLICATE
response = cv2.medianBlur(image, kernel_size)
```

### 高斯滤波

```python
response = cv2.GaussianBlur(image, (kx, ky), sigma)
```

### 双边滤波

```python
response = cv2.bilateralFilter(image, kernel_size, color_sigma, space_sigma)
```