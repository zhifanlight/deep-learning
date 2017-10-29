<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 傅立叶变换

&nbsp;

## 背景介绍

- 任何连续测量的时序或信号，都可以表示为不同频率的正弦波信号的无限叠加

- 图像频率是图像灰度变化剧烈程度的指标：变化越剧烈，频率越高。低频分量决定图像基本结构，高频分量决定图像边缘和细节


- 在频率域，振幅决定了该分量对原始信号的重要性。低频分量的振幅较大，高频分量的振幅较小。在傅立叶谱中，振幅越大，亮度越高

- 大自然中的各种信号的大部分信息都集中在低频，人眼对低频信号更敏感

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

## 卷积定理

- 函数卷积的傅立叶变换等于各自傅立叶变换的乘积：

	$$F(f * g) = F(f) \cdot F(g)$$
	
	- 逆向傅立叶变换后，图像整体向右下角平移，距离为卷积核的一半

- 函数相关的傅立叶变换等于前者傅立叶变换乘上后者傅立叶变换的共轭：

	$$F(f \otimes g) = F(f) \cdot F_{C}(g)$$
	
	- 逆向傅立叶变换后，图像整体向左上角平移，距离为卷积核的一半

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

- 卷积定理（卷积）

	```
	# straight convolution
	straight = cv2.filter2D(frame, cv2.CV_32F, cv2.flip(kernel, -1))
	
	# dft acceleration
	opt_height = cv2.getOptimalDFTSize(frame_height + kernel_height - 1)
	opt_width = cv2.getOptimalDFTSize(frame_width + kernel_width - 1)

	# copy data and dft
	extend_frame[:frame_height, :frame_width] = frame
	extend_kernel[:kernel_height, :kernel_width] = kernel
	dft_frame = cv2.dft(extend_frame, flags=cv2.DFT_COMPLEX_OUTPUT)
	dft_kernel = cv2.dft(extend_kernel, flags=cv2.DFT_COMPLEX_OUTPUT)
	
	# complex multiply
	dft_mul = cv2.mulSpectrums(dft_frame, dft_kernel, 0, conjB=False)
	
	# idft
	idft_frame = cv2.idft(dft_mul, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
	
	# shift image
	response[:, :] = idft_frame[half: frame_height + half, half: frame_width + half]
	```

- 卷积定理（相关）

	```
	# straight correlation
	straight = cv2.filter2D(frame, cv2.CV_32F, kernel)
	
	# dft acceleration
	opt_height = cv2.getOptimalDFTSize(frame_height + kernel_height - 1)
	opt_width = cv2.getOptimalDFTSize(frame_width + kernel_width - 1)

	# copy data and dft
	extend_frame[:frame_height, :frame_width] = frame
	extend_kernel[:kernel_height, :kernel_width] = kernel
	dft_frame = cv2.dft(extend_frame, flags=cv2.DFT_COMPLEX_OUTPUT)
	dft_kernel = cv2.dft(extend_kernel, flags=cv2.DFT_COMPLEX_OUTPUT)
	
	# complex multiply with matrix B conjugated
	dft_mul = cv2.mulSpectrums(dft_frame, dft_kernel, 0, conjB=True)
	
	# idft
	idft_frame = cv2.idft(dft_mul, flags=cv2.DFT_SCALE | cv2.DFT_REAL_OUTPUT)
	
	# shift image
	response[half:, half:] = idft_frame[: frame_height - half, : frame_width - half]
	response[: half, : half] = idft_frame[-half:, -half:]
	response[: half, half:] = idft_frame[-half:, : frame_width - half]
	response[half:, : half] = idft_frame[: frame_height - half, -half:]
	```