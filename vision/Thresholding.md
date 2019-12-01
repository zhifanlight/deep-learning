# 图像二值化

## 原理分析

- 通过特定的阈值，将灰度图转化成二值图像

## 常用方法

### 双峰法

- 在一些简单图像中，目标与背景在直方图各形成一个波峰

- 可以将两个波峰间的波谷灰度值作为阈值，实现目标和背景的分割

### 迭代法

- 随机初始化阈值 $T$

- 根据阈值 $T$ 将图像分割成 $P_{1}$ 和 $P_{2}$

  - $P_{1}$ 由灰度值小于 $T$ 的像素点组成

  - $P_{2}$ 由灰度值大于 $T$ 的像素点组成

- 计算 $P_{1}$ 部分的平均灰度值 $V_{1}$，$P_{2}$ 部分的平均灰度值 $V_{2}$

- 将 $\frac{V_{1} + V_{2}}{2}$ 作为新的阈值，重复上述两步，直到新、旧阈值足够接近

### $\mathrm{OTSU}$ 法（大津法）

- $\omega_{0}, \ \omega_{1}$ 分别为小于、大于阈值 $T$ 的像素点比例，$\mu_{0}, \ \mu_{1}$ 分别是这两部分像素点的平均灰度值，$\mu_{T}$ 是整幅图像的平均灰度值，即 $\mu_{T} = \omega_{0} \mu_{0} + \omega_{1} \mu_{1}$

- 组间方差：$\omega_{0} \left( \mu_{0} - \mu_{T} \right)^{2} + \omega_{1} \left( \mu_{1} - \mu_{T} \right)^{2} = \omega_{0} \omega_{1} \left( \mu_{0} - \mu_{1} \right)^{2}$

- 最大化组间方差：迭代所有的灰度值，选择组间方差最大的灰度值作为阈值

## $\mathrm{Python}$ 实现

### $\mathrm{OTSU}$ 法

```python
threshold, binary = cv2.threshold(frame, thresh, max_value, type=cv2.THRESH_TYPE)
```

#### 指定阈值

```python
type=cv2.THRESH_BINARY        # value = value > thresh ? max_value : 0 
type=cv2.THRESH_BINARY_INV    # value = value > thresh ? 0 : max_value
type=cv2.THRESH_TOZERO        # value = value > thresh ? value : 0
type=cv2.THRESH_TOZERO_INV    # value = value > thresh ? 0 : value
type=cv2.THRESH_TRUNC         # value = value > thresh ? thresh : value
```