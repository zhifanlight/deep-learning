# 直方图

## 彩色直方图

- 将彩色直方图 $3$ 个通道分离，分别计算其灰度直方图

## 直方图归一化

- 直方图每个 $\mathrm{bin}$ 对应的数值为像素值出现概率，概率之和为 $1$

- 可以消除图片尺度变化带来的影响

## 直方图拉伸

- 把灰度直方图从某个密集区间，线性（或非线性）扩展到更大的区间，扩大前景与背景差异，增强对比度

- 计算过程

  - 设定 $\mathrm{bin}$ 中最低像素数量阈值，寻找直方图中满足条件的最小、最大索引 $\mathrm{min}$、$\mathrm{max}$

  - 计算直方图查找表： $\mathrm{LUT[idx]} = 255 \cdot \frac {\mathrm{idx - min}} {\mathrm{max - min}}$

  - 对于原始图像中的每个像素值 $\mathrm{value}$，将其更新为 $\mathrm{LUT[value]}$

## 直方图均衡化

- 通过累积分布函数，把灰度直方图从某个密集区间，非线性扩展成全部灰度范围内的均匀分布，增强对比度

- 在像素值比较密集的区间，其累积概率值变化较大，均衡化后，这些区间会变得离散，相邻像素值得以分离

- 计算过程

  - 计算直方图中每个像素出现的概率分布 $P$

  - 计算 $P$ 对应的累积分布 $C$

  - 对于原始图像中的每个像素值 $\mathrm{value}$，将其更新为 $255 \cdot C[\mathrm{value}]$

## 直方图对比

- 比较归一化后直方图相似性

  - 相关性：值越大，相似性越高

    $$
    d \left( H_{1}, \ H_{2} \right) = \frac{\sum_{p} \left[ H_{1} \left( p \right) - \bar {H_{1}} \right] \left[ H_{2} \left( p \right) - \bar {H_{2}} \right]} {\sqrt{ \sum_{p} \left[ H_{1} \left( p \right) - \bar {H_{1}} \right]^{2} \sum_{p} \left[ H_{2} \left( p \right) - \bar {H_{2}} \right]^{2} }}
    $$

  - 相交性：值越大，相似性越高

    $$
    d \left( H_{1}, \ H_{2} \right) = \sum_{p} \min \left( H_{1} \left( p \right), \ H_{2} \left( p \right) \right)
    $$

  - 卡方距离：值越小，相似性越高

    $$
    d \left( H_{1}, \ H_{2} \right) = \sum_{p} \frac { \left[ H_{1} \left( p \right) - H_{2} \left( p \right) \right]^{2}} {H_{1} \left( p \right)}
    $$

  - 巴氏距离：值越小，相似性越高（ $ N $ 为 $\mathrm{bin}$ 个数 ）

    $$
    d \left( H_{1}, \ H_{2} \right) = \sqrt{1 - \frac{1}{\sqrt{\bar {H_{1}} \cdot \bar {H_{2}} \cdot N^{2}}} \sum_{p} \sqrt{H_{1} \left( p \right) \cdot H_{2} \left( p \right)}}
    $$

## $\mathrm{Python}$ 实现

### 灰度直方图

```python
hist = cv2.calcHist(images=[image], channels=[0], mask=None, histSize=[256], ranges=[0, 255])
```

### 直方图归一化

```python
hist = hist / image.size
```

### 直方图拉伸

```python
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

```python
image = cv2.equalizeHist(image)
```

### 直方图对比

```python
similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_INTERSECT)

similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CHISQR)

similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_BHATTACHARYYA)
```