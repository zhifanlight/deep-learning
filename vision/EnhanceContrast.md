<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 对比度增强

&nbsp;

## 背景介绍

- 增强图像中的有用信息，改善图像视觉效果

- 有目的地强调图像的整体或局部特性，将原来不清晰的图像变得清晰

- 强调某些感兴趣的特征，抑制不感兴趣的特征，扩大图像中不同物体特征之间的差别

- 分为两大类：

	- 间接对比度增强

		- 通过操作图像直方图来增强对比度

		- 直方图拉伸

		- 直方图均衡化

	- 直接对比度增强

		- 直接在操作原始图像来增强对比度

		- \\( Log \\) 变换

		- \\( Gamma \\) 变换

## 间接对比度增强

- 直方图拉伸

	- 把灰度直方图从某个密集区间，线性（或非线性）扩展到更大的区间，扩大前景与背景差异，增强对比度

	- 计算过程

		- 设定 \\( bin \\) 中最低像素数量阈值，寻找直方图中满足条件的最小、最大索引 \\( min \\)、\\( max \\)

		- 计算直方图查找表： \\( LUT[idx] = 255 * \frac {idx - min} {max - min} \\)

		- 对于原始图像中的每个像素值 \\( value \\)，将其更新为 \\( LUT[value] \\)

- 直方图均衡化

	- 通过累积分布函数，把灰度直方图从某个密集区间，非线性扩展成全部灰度范围内的均匀分布，增强对比度

	- 在像素值比较密集的区间，其累积概率值变化较大，均衡化后，这些区间会变得离散，相邻像素值得以分离

	- 计算过程

		- 计算直方图中每个像素出现的概率分布 \\( P \\)
	
		- 计算 \\( P \\) 对应的累积分布 \\( C \\) 
	
		- 对于原始图像中的每个像素值 \\( value \\)，将其更新为 \\( 255 * C[value] \\)

	- 对于背景、前景都太亮或太暗的图像非常有用

	- 已知均衡化函数，可逆向恢复原始直方图

	- 如果直方图像素分布过于集中，可能会增加背景对比度而减少有用信号对比度

## 直接对比度增强

- \\( Log \\) 变换

	- \\( s = c * log _ {v+1} (1 + v * r) \\)，其中 \\( r \in [0, 1] \\)，而 \\( c, v \\) 为常数

	- \\( Log \\) 变换可以将图像的低灰度值部分扩展，显示低灰度部分更多的细节；将图像的高灰度值部分压缩，减少高灰度值部分的细节，从而增强图像低灰度部分
	
	- 由于 \\( s \approx c + c * log _ {v+1} (r) \\)，在区间 \\( [0, 1] \\) 内，底数越大，图像越陡峭 ，对低灰度值部分的扩展越强，对高灰度值部分的压缩也越强

	![](images/log.png)

- \\( Gamma \\) 变换

	- \\( s = c * r ^ {\gamma} \\)，其中 \\( r \in [0, 1] \\)，而 \\( c, v \\) 为常数

	- 当 \\( \gamma \\) 值确定时，\\( s \\) 是 \\( r \\) 的幂函数，且以 1 为分界线

		- 值越小，对图像低灰度值部分的扩展作用越明显

		- 值越大，对图像高灰度值部分的扩展作用越明显

	- 通过不同的 \\( \gamma \\) 值，可以达到增强低灰度或高灰度部分细节的作用
	
	![](images/gamma.png) 

&nbsp;

## Python 实现

- 直方图拉伸

	```
	lut = numpy.zeros((256, 1), numpy.uint8)

	for idx, elem in enumerate(hist):
		if idx < min_idx:
			lut[idx] = 0
		elif idx > max_idx:
			lut[idx] = 255
		else:
			lut[idx] = int(round(255.0 * (idx - min_idx) / (max_idx - min_idx)))

	image = cv2.LUT(image, lut)
	```

- 直方图均衡化（只处理单通道）

	```
	image = cv2.equalizeHist(image)
	``` 
	
- \\( Log \\) 变换

	```
	image = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	response = c * numpy.log2(1 + v * image) / numpy.log2(v + 1)
	```

- \\( Gamma \\) 变换

	```
	image = cv2.normalize(image, None, 0.0, 1.0, cv2.NORM_MINMAX, dtype=cv2.CV_32F)
	response = c * numpy.power(image, gamma)
	```