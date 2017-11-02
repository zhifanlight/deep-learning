<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 边缘检测

&nbsp;

## Sobel

- 一阶导数极值处定义为边缘

- 分别在 x 和 y 两个方向上进行检测：

	$$ G\_{x} = \\left[ \\begin{matrix} -1 & 0 & +1 \\\\ -2 & 0 & +2 \\\\ -1 & 0 & +1 \\end{matrix} \\right], \quad G\_{y} = \\left[ \\begin{matrix} -1 & -2 & -1 \\\\ 0 & 0 & 0 \\\\ +1 & +2 & +1 \\end{matrix} \\right] $$

- 然后将结果合并：

	$$ G = \sqrt{G\_{x}^{2} + G\_{y}^{2}} \quad or \quad G = |G\_{x}| + |G\_{y}| $$

- 产生的边缘有强弱；由于结合了高斯平滑，抗噪性强

## Scharr

- 一阶导数极值处定义为边缘

- 分别在 x 和 y 两个方向上进行检测：

	$$ G\_{x} = \\left[ \\begin{matrix} -3 & 0 & +3 \\\\ -10 & 0 & +10 \\\\ -3 & 0 & +3 \\end{matrix} \\right], \quad G\_{y} = \\left[ \\begin{matrix} -3 & -10 & -3 \\\\ 0 & 0 & 0 \\\\ +3 & +10 & +3 \\end{matrix} \\right] $$

- 然后将结果合并：

	$$ G = \sqrt{G\_{x}^{2} + G\_{y}^{2}} \quad or \quad G = |G\_{x}| + |G\_{y}| $$

- 与 Sobel 原理相同，但效果更好

## Canny

- 一阶导数极值处定义为边缘

- 使用 Sobel 等边缘检测器进行边缘检测，计算幅值和方向：

	$$ G = \sqrt{G\_{x}^{2} + G\_{y}^{2}} \ , \quad \theta = arctan\left(\frac{G\_{y}}{G\_{x}}\right) $$
	
- 使用滞后阈值：低阈值 \\(T\_{1}\\)、高阈值 \\(T\_{2}\\)：

	- 如果某像素位置的幅值超过 \\(T\_{2}\\)，该像素被保留为边缘像素

	- 如果某像素位置的幅值小于 \\(T\_{1}\\)，该像素被排除

	- 如果某像素位置的幅值介于两者之间，则仅当该像素连接到高于 \\(T\_{2}\\) 的像素时才被保留

	- \\(T\_{1}: T\_{2}\\) 的通常介于 2:1 到 3:1

- 产生的边缘较细，可能只有一个像素，无强弱之分

## Laplacian

- 一阶导数极值处，二阶导数为零；可以检测出二阶导数的零点作为边缘

- 4 向 Laplacian 卷积核 \\( G = \\left[ \\begin{matrix} 0 & 1 & 0 \\\\ 1 & -4 & 1 \\\\ 0 & 1 & 0 \\end{matrix} \\right] \\) 推导：
	
	$$
	\begin{align\*}
	\frac{\partial^{2} f}{\partial x^{2}} &= \left[ f(x + 1, y) - f(x, y) \right] - \left[ f(x, y) - f(x - 1, y) \right] \newline
	&= f(x + 1, y) + f(x - 1, y) - 2 * f(x, y) \newline \newline
	\frac{\partial^{2} f}{\partial y^{2}} &= \left[ f(x, y + 1) - f(x, y) \right] - \left[ f(x, y) - f(x, y - 1) \right] \newline
	&= f(x, y + 1) + f(x, y - 1) - 2 * f(x, y) \newline \newline
	\nabla^{2}f &= \frac{\partial^{2} f}{\partial x^{2}} + \frac{\partial^{2} f}{\partial y^{2}} \newline
	&= f(x - 1, y) + f(x + 1, y) + f(x, y - 1) + f(x, y + 1) - 4 * f(x, y) \newline
	\end{align\*}
	$$

- 同理可得，8 向 Laplacian 卷积核 \\( G = \\left[ \\begin{matrix} 1 & 1 & 1 \\\\ 1 & -8 & 1 \\\\ 1 & 1 & 1 \\end{matrix} \\right] \\)

- 在图像中较暗的区域中出现了亮点，Laplacian 运算会使该点变得更亮

- 对噪声比较敏感；一般先平滑，再计算边缘

&nbsp;

## Python 实现

### Sobel

```
xx = cv2.Sobel(image, cv2.CV_16S, 1, 0)
yy = cv2.Sobel(image, cv2.CV_16S, 0, 1)
abs_x = cv2.convertScaleAbs(xx)
abs_y = cv2.convertScaleAbs(yy)
response = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
```

### Scharr

```
xx = cv2.Scharr(image, cv2.CV_16S, 1, 0)
yy = cv2.Scharr(image, cv2.CV_16S, 0, 1)
abs_x = cv2.convertScaleAbs(xx)
abs_y = cv2.convertScaleAbs(yy)
response = cv2.addWeighted(abs_x, 0.5, abs_y, 0.5, 0)
```

### Canny

```
response = cv2.Canny(image, thre1, thre2)
```

### Laplacian（4 向）

```
response = cv2.Laplacian(image, cv2.CV_16S)
response = cv2.convertScaleAbs(response)
```