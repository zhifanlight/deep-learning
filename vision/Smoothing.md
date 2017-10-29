<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 图像平滑

&nbsp;

## 原理分析

- 消除或减少噪声影响

- 用每个像素周围、固定范围内的像素值计算当前像素值

- 常见方法

	- 均值滤波

		- 用邻域内像素点的平均值作为当前点像素值

		- 对噪音图像敏感，即使有少量噪声点存在较大差异，也会导致平均值的明显波动
	
	- 中值滤波

		- 用邻域内像素点的中值作为当前点像素值

		- 滤波器为正方形，且尺寸只能为奇数

		- 可以较好的消除椒盐噪声，较好的保存图像边缘信息

	- 高斯滤波

		- 由于图像的连续性，距离越近的点权重越大，距离越远的点权重越小
		
		- 通过二维高斯分布计算高斯核：\\( G(x, y) = \frac {1} { 2 \pi \sigma \_{1} \sigma \_{2} } e ^ { - (\frac {(x - \mu \_{x} ) ^ {2}} {2 \sigma \_{1} ^ {2}} + \frac {(y - \mu \_{y}) ^ {2}} {2 \sigma \_{2} ^ {2}}) } \\)

		- 对高斯核进行缩放，使所有 cell 和为 1，并用缩放后的高斯核对图像进行卷积

		- 高斯核的尺寸只能为奇数。\\( \mu \_{x}, \mu \_{y} = \frac {x\\\_size} {2}, \frac {y\\\_size} {2} \\)

	- 双边滤波

		- 同时考虑空间距离与像素差值的影响，可以在保留边缘信息的同时去除图片噪声

		- \\( k \\) 为归一化系数；\\( p(x, y) \\) 表示 \\( (x, y) \\) 点的像素值；\\( s(a, b, x, y) \\) 表示 \\( (a, b) \\) 和 \\( (x, y) \\) 两个点空间距离的高斯值；\\( c(x, y) \\) 表示 \\( x, y \\) 两种像素值距离的高斯值

		- \\( G(x, y) = k(x, y) ^ {-1} \\iint p(a, b) \cdot s(a, b, x, y) \cdot c(p(a, b), p(x, y)) dadb \\)

		- \\( s(a, b, x, y) = e ^ { - \frac {(a - x) ^ {2} + (b - y) ^ {2}} {2 \sigma \_{s} ^ {2}} } \\)，\\( c(x, y) = e ^ {- \frac {(x - y) ^ 2} {2 \sigma \_{c} ^{2}}} \\)

		- \\( k(x, y) = \iint s(a, b, x, y) \cdot c(p(a, b), p(x, y)) dadb \\)

&nbsp;

## Python 实现

- 均值滤波

	```
	response = cv2.blur(image, (x_size, y_size))
	``` 

- 中值滤波

	```
	borderType = cv2.BORDER_REPLICATE
	response = cv2.medianBlur(image, kernel_size)
	``` 

- 高斯滤波

	```
	response = cv2.GaussianBlur(image, (x_size, y_size), sigma)
	```
	
- 双边滤波

	```
	response = cv2.bilateralFilter(image, kernel_size, color_sigma, space_sigma)
	```