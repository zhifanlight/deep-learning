<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 图像二值化

## 原理分析

- 通过特定的阈值，将灰度图转化成二值图像

## 常用方法

### 双峰法

- 在一些简单图像中，目标与背景在直方图各形成一个波峰

- 可以将两个波峰间的波谷灰度值作为阈值，实现目标和背景的分割

### 迭代法

- 随机初始化阈值 T

- 根据阈值 T 将图像分割成 \\(P\_{1}\\) 和 \\(P\_{2}\\)

	- \\(P\_{1}\\) 由灰度值小于 T 的像素点组成

	- \\(P\_{2}\\) 由灰度值大于 T 的像素点组成

- 计算 \\(P\_{1}\\) 部分的平均灰度值 \\(V\_{1}\\)，\\(P\_{2}\\) 部分的平均灰度值 \\(V\_{2}\\)

- 将 \\(\frac{V\_{1} + V\_{2}}{2}\\) 作为新的阈值，重复上述两步，直到新、旧阈值足够接近

### OTSU 法（大津法）

- \\(\omega\_{0}, \omega\_{1}\\) 分别为小于、大于阈值 T 的像素点比例，\\(\mu\_{0}, \mu\_{1}\\) 分别是这两部分像素点的平均灰度值，\\(\mu\_{T}\\) 是整幅图像的平均灰度值，即 \\(\mu\_{T} = \omega\_{0}\mu\_{0} + \omega\_{1}\mu\_{1}\\)

- 组间方差：\\(\omega\_{0} (\mu\_{0} - \mu\_{T})^{2} + \omega\_{1} (\mu\_{1} - \mu\_{T})^{2} = \omega\_{0} \omega\_{1} (\mu\_{0} - \mu\_{1}) ^{2}\\)

- 最大化组间方差：迭代所有的灰度值，选择组间方差最大的灰度值作为阈值

## Python 实现

### OTSU 法

```
threshold, binary = cv2.threshold(frame, thresh, max_value, type=cv2.THRESH_TYPE)
```

#### 指定阈值

```
type=cv2.THRESH_BINARY        # value = value > thresh ? max_value : 0 
type=cv2.THRESH_BINARY_INV    # value = value > thresh ? 0 : max_value
type=cv2.THRESH_TOZERO        # value = value > thresh ? value : 0
type=cv2.THRESH_TOZERO_INV    # value = value > thresh ? 0 : value
type=cv2.THRESH_TRUNC         # value = value > thresh ? thresh : value
```