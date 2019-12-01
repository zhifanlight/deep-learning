# 边缘检测

## $\mathrm{Sobel}$

- 一阶导数极值处定义为边缘

- 分别在 $x$ 和 $y$ 两个方向上进行检测：

  $$
  G_{x} = \left[ \begin{matrix} -1 & 0 & +1 \\ -2 & 0 & +2 \\ -1 & 0 & +1 \end{matrix} \right], \quad G_{y} = \left[ \begin{matrix} -1 & -2 & -1 \\ 0 & 0 & 0 \\ +1 & +2 & +1 \end{matrix} \right]
  $$

- 然后将结果合并：

  $$
  G = \sqrt{G_{x}^{2} + G_{y}^{2}} \quad \mathrm{or} \quad G \approx |G_{x}| + |G_{y}|
  $$

- 产生的边缘有强弱；由于结合了高斯平滑，抗噪性强

## $\mathrm{Scharr}$

- 一阶导数极值处定义为边缘

- 分别在 $x$ 和 $y$ 两个方向上进行检测：

  $$
  G_{x} = \left[ \begin{matrix} -3 & 0 & +3 \\ -10 & 0 & +10 \\ -3 & 0 & +3 \end{matrix} \right], \quad G_{y} = \left[ \begin{matrix} -3 & -10 & -3 \\ 0 & 0 & 0 \\ +3 & +10 & +3 \end{matrix} \right]
  $$

- 然后将结果合并：

  $$
  G = \sqrt{G_{x}^{2} + G_{y}^{2}} \quad \mathrm{or} \quad G \approx |G_{x}| + |G_{y}|
  $$

- 与 $\mathrm{Sobel}$ 原理相同，但效果更好

## $\mathrm{Canny}$

- 一阶导数极值处定义为边缘

- 使用 $\mathrm{Sobel}$ 等边缘检测器进行边缘检测，计算梯度大小和方向：

  $$
  G = \sqrt{G_{x}^{2} + G_{y}^{2}} \ , \quad \theta = \arctan \left( \frac{G_{y}}{G_{x}} \right)
  $$

- 设置低阈值 $T_{1}$、高阈值 $T_{2}$：

  - 如果某像素的梯度值超过 $T_{2}$，该像素被保留为边缘像素

  - 如果某像素的梯度值小于 $T_{1}$，该像素被排除

  - 如果某像素的梯度值介于两者之间，则仅当该像素连接到高于 $T_{2}$ 的像素时才被保留

  - $T_{1}: \ T_{2}$ 的通常介于 $2: \ 1$ 到 $3: \ 1$

- 产生的边缘较细，可能只有一个像素，无强弱之分

## $\mathrm{Laplacian}$

- 一阶导数极值处，二阶导数为零

  - 通过寻找二阶导数的零点，进行边缘检测

- $4$ 邻域 $\mathrm{Laplacian}$ 卷积核 $ G = \left[ \begin{matrix} 0 & 1 & 0 \\ 1 & -4 & 1 \\ 0 & 1 & 0 \end{matrix} \right] $ 推导：

  $$
  \begin{aligned}
  \frac{\partial^{2} f}{\partial x^{2}} &= \left[ f \left( x + 1, \ y \right) - f \left( x, \ y \right) \right] - \left[ f \left( x, \ y \right) - f \left( x - 1, \ y \right) \right] \newline
  &= f \left( x + 1, \ y \right) + f \left( x - 1, \ y \right) - 2 \cdot f \left( x, \ y \right) \newline \newline
  \frac{\partial^{2} f}{\partial y^{2}} &= \left[ f \left( x, \ y + 1 \right) - f \left( x, \ y \right) \right] - \left[ f \left( x, \ y \right) - f \left( x, \ y - 1 \right) \right] \newline
  &= f \left( x, \ y + 1 \right) + f \left( x, \ y - 1 \right) - 2 \cdot f \left( x, \ y \right) \newline \newline
  \nabla^{2} f &= \frac{\partial^{2} f}{\partial x^{2}} + \frac{\partial^{2} f}{\partial y^{2}} \newline
  &= f \left( x - 1, \ y \right) + f \left( x + 1, \ y \right) + f \left( x, \ y - 1 \right) + f \left( x, \ y + 1 \right) - 4 \cdot f \left( x, \ y \right) \newline
  \end{aligned}
  $$

- 同理可得，$8$ 邻域 $\mathrm{Laplacian}$ 卷积核 $ G = \left[ \begin{matrix} 1 & 1 & 1 \\ 1 & -8 & 1 \\ 1 & 1 & 1 \end{matrix} \right] $

- 在图像中较暗的区域中出现了亮点，$\mathrm{Laplacian}$ 运算会使该点变得更亮

- 对噪声比较敏感；一般先平滑，再计算边缘

## $\mathrm{Python}$ 实现

### $\mathrm{Sobel}$

```python
xx = cv2.Sobel(image, cv2.CV_16S, 1, 0)
yy = cv2.Sobel(image, cv2.CV_16S, 0, 1)
abs_x = cv2.convertScaleAbs(xx)
abs_y = cv2.convertScaleAbs(yy)
response = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
```

### $\mathrm{Scharr}$

```python
xx = cv2.Scharr(image, cv2.CV_16S, 1, 0)
yy = cv2.Scharr(image, cv2.CV_16S, 0, 1)
abs_x = cv2.convertScaleAbs(xx)
abs_y = cv2.convertScaleAbs(yy)
response = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
```

### $\mathrm{Canny}$

```python
response = cv2.Canny(image, thre1, thre2)
```

### $\mathrm{Laplacian}$（$4$ 邻域）

```python
response = cv2.Laplacian(image, cv2.CV_16S)
response = cv2.convertScaleAbs(response)
```