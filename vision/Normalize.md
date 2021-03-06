# 归一化

## 原理分析

- 将原数据从原始区间转化到新的区间

- 对于彩色图像，每个通道单独处理

## 常见方法

### 线性

- 已知原数据最值分别为 $\mathrm{min}$、$\mathrm{max}$

- 所有数据平移或缩放到区间 $\left[ a, \ b \right]$

  $$
  new = \frac {\mathrm{old} - \mathrm{min}} {\mathrm{max} - \mathrm{min}} \cdot \left( b - a \right) + a
  $$

### $\mathrm{L_{1}}$ 范数

- 已知原数据绝对值之和为 $S$，给定范数值 $L$

- 处理后数据的绝对值之和为 $L$，所有数据等比例缩放

  $$
  \mathrm{new} = \mathrm{old} \cdot \frac {L} {S}
  $$

### $\mathrm{L_{2}}$ 范数

- 已知原数据平方和为 $S$，给定范数值 $L$

- 处理后数据的平方和为 $L^{2}$，所有数据等比例缩放

  $$
  \mathrm{new} = \mathrm{old} \cdot \sqrt{\frac{L^{2}}{S}}
  $$

### 无穷范数

- 已知原数据绝对值最大值为 $M$，给定范数值 $L$

- 处理后数据的绝对值最大值为 $L$，所有数据等比例缩放

  $$
  \mathrm{new} = \mathrm{old} \cdot \frac {L} {M}
  $$

## $\mathrm{Python}$ 实现

### 线性

```python
cv2.normalize(array, None, alpha=a, beta=b, norm_type=cv2.NORM_MINMAX)
```

### $\mathrm{L_{1}}$ 范数

```python
cv2.normalize(array, None, alpha=L, norm_type=cv2.NORM_L1)
```

### $\mathrm{L_{2}}$ 范数

```python
cv2.normalize(array, None, alpha=L, norm_type=cv2.NORM_L2)
```

### 无穷范数

```python
cv2.normalize(array, None, alpha=L, norm_type=cv2.NORM_INF)
```