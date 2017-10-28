<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 傅立叶变换

&nbsp;

## 背景介绍

- 任何连续测量的时序或信号，都可以表示为不同频率的正弦波信号的无限叠加

- 图像频率是图像灰度变化剧烈程度的指标：变化越剧烈，频率越高。低频分量决定图像基本结构，高频分量决定图像边缘和细节


- 在频率域，振幅决定了该分量对原始信号的重要性。低频分量的振幅较大，高频分量的振幅较小。在傅立叶谱中，振幅越大，亮度越高

- 大自然中的各种信号的大部分信息都集中在低频，人眼对低频更敏感

## 原理分析

- 假设 \\(f, F\\) 分别表示空间域、频率域上的信号，宽、高分别为 \\(M, N\\)

	- 正向傅立叶变换从空间域变换到频率域：

		$$ F(u,v) = \sum\_{x=0}^{M-1} \sum\_{y=0}^{N-1} f(x,y) \ e^{-j2\pi (\frac{ux}{M} + \frac{vy}{N})}$$

	- 逆向傅立叶变换从频率域变换到空间域：

		$$ f(x,y) = \frac{1}{MN} \sum\_{u=0}^{M-1} \sum\_{v=0}^{N-1} F(u,v) \ e^{j2\pi (\frac{ux}{M} + \frac{vy}{N})}$$
		
	- 欧拉公式：
	
		$$ e^{jx} = cos(x) + j\ sin(x) $$ 

	- 频谱直流分量（图像平均灰度）：
	
		$$ F(0, 0) = \frac{1}{MN} \sum\_{x=0}^{M-1} \sum\_{y=0}^{N-1} f(x,y) $$

- 二维傅立叶变换是两个一维傅立叶变换叠加：先按行进行变换，再按列进行变换

- 正向傅立叶变换时，低频分量在边缘，高频分量在中心。通常要对傅立叶谱进行中心化

- 逆向傅立叶变换时，需要先进行逆中心化，再进行逆变换

&nbsp;

## Python 实现
 
- 正向傅立叶变换

	```
	transform = cv2.dft(numpy.float32(image), flags=cv2.DFT_COMPLEX_OUTPUT)
	transform = numpy.fft.fftshift(transform)
	magnitude = cv2.log(1.0 + cv2.magnitude(transform[:, :, 0], transform[:, :, 1]))
	response = cv2.normalize(magnitude, None, 0.0, 255.0, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U)
	``` 

- 逆向傅立叶变换

	```
	transform = numpy.fft.ifftshift(transform)
	transform = cv2.idft(transform, flags=cv2.DFT_SCALE)
	magnitude = cv2.magnitude(transform[:, :, 0], transform[:, :, 1])
	response = numpy.uint8(numpy.round(magnitude))
	``` 