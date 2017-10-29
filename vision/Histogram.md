<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 直方图

&nbsp;

## 彩色直方图

- 将彩色直方图 3 个通道分离，分别计算其灰度直方图

## 直方图归一化

- 直方图每个 \\( bin \\) 对应的数值为像素值出现概率，概率之和为 1

- 可以消除图片尺度变化带来的影响

## 直方图拉伸

- 把灰度直方图从某个密集区间，线性（或非线性）扩展到更大的区间，扩大前景与背景差异，增强对比度

- 计算过程

	- 设定 \\( bin \\) 中最低像素数量阈值，寻找直方图中满足条件的最小、最大索引 \\( min \\)、\\( max \\)

	- 计算直方图查找表： \\( LUT[idx] = 255 \cdot \frac {idx - min} {max - min} \\)

	- 对于原始图像中的每个像素值 \\( value \\)，将其更新为 \\( LUT[value] \\)

## 直方图均衡化

- 通过累积分布函数，把灰度直方图从某个密集区间，非线性扩展成全部灰度范围内的均匀分布，增强对比度

- 在像素值比较密集的区间，其累积概率值变化较大，均衡化后，这些区间会变得离散，相邻像素值得以分离

- 计算过程

	- 计算直方图中每个像素出现的概率分布 \\( P \\)
	
	- 计算 \\( P \\) 对应的累积分布 \\( C \\) 
	
	- 对于原始图像中的每个像素值 \\( value \\)，将其更新为 \\( 255 \cdot C[value] \\)

## 直方图匹配

- 把灰度直方图变成某一特定形状的直方图，有目的地增强某个灰度区间

- 直方图均衡化的扩展，可以得到任意形状的灰度直方图

- 处理后的图像，具有与目标图像相似的颜色特征

- 数学原理

	- 原始直方图为 \\( r \\)，均衡化结果为 \\( s \\)，变化函数为 \\( s = T(r) \\)

	- 目标直方图为 \\( z \\)，均衡化结果为 \\( v \\)，变换函数为 \\( v = G(z) \\)

	- 匹配目标是原始直方图、目标直方图尽可能接近，即 \\( s \_{k} = v \_{m} \\)

- 计算过程

	- 分别计算原始图像、目标图像的累积分布直方图 \\( R \\) 和 \\( Z \\)

	- 对于 \\( R \\) 的每个 \\( bin \\)，在 \\( Z \\) 中找到与其概率值最接近的 \\( bin \\)，记录 \\( bin \\) 间映射关系
	
	- 根据以上映射关系，对整张图像进行映射

## 直方图对比

- 比较归一化后直方图相似性

	- 相关性：值越大，相似性越高
	
	$$ d(H _{1}, H _{2}) = \frac {\sum _{p} [H _{1} (p) - \bar {H _{1}}] [(H _{2} (p) - \bar {H _{2}}]} {\sqrt { \sum _{p} [H _{1} (p) - \bar {H _{1}}] ^{2} \sum _{p} [H _{2} (p) - \bar {H _{2}}] ^{2} }} $$
	
	- 相交性：值越大，相似性越高
	
	$$ d(H _{1}, H _{2}) = \sum _{p} min(H _{1} (p), H _{2} (p)) $$

	- 卡方距离：值越小，相似性越高
	
	$$ d(H _{1}, H _{2}) = \sum _{p} \frac {[H _{1} (p) - H _{2} (p)] ^{2}} {H _{1} (p)} $$

	- 巴氏距离：值越小，相似性越高（ \\( N \\) 为 \\( bin \\) 个数 ）
	
	$$ d(H _{1}, H _{2}) = \sqrt {1 - \frac {1} {\sqrt {\bar {H _{1}} \cdot \bar {H _{2}} \cdot N ^{2}}} \sum _{p} \sqrt {H _{1} (p) \cdot H _{2} (p)}} $$

&nbsp;

## Python 实现

### 灰度直方图

```
hist = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
```

### 直方图归一化

```
hist = hist / image.size
```

### 直方图拉伸

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

### 直方图均衡（只处理单通道）

```
image = cv2.equalizeHist(image)
```

### 直方图匹配

```
origin_hist = get_cumulative_histogram(origin_hist)
target_hist = get_cumulative_histogram(target_hist)

diff = numpy.zeros((256, 256))
for i in range(256):
	for j in range(256):
		diff[i][j] = abs(origin_hist[i] - target_hist[j])
			
lut = numpy.zeros((256, ))
for i in range(256):
	lut[i] = index_of_minimum(diff[i][:])
	
image = cv2.LUT(origin, lut).astype(numpy.uint8)
```

### 直方图对比

```
similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
# ```