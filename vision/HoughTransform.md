# 霍夫变换

## 标准霍夫线变换

### 原理分析

- 在笛卡尔坐标系中，除了与 $x$ 轴垂直的直线，大部分直线 $y = kx + b$ 可由斜率、截距 $\left( k, \ b \right)$ 表示

- 而在极坐标系中，包括与坐标轴垂直的直线在内，所有直线可由极径、极角 $\left( r, \ \theta \right)$ 表示

<center>
<img src="images/hough_polar_coords.png"/>
</center>

  - 红色直线可表示为：$y = \left( -\frac{\cos \theta}{\sin \theta} \right) x + \left( \frac{r}{\sin \theta} \right)$

  - 进一步化简得到：$r = x \cdot \cos \theta + y \cdot \sin \theta$

  - 一般来说，经过点 $\left( x_{0}, \ y_{0} \right)$ 的一族直线可定义为：$r = x_{0} \cdot \cos \theta + y_{0} \cdot \sin \theta$，形状为正弦曲线；而曲线上的每个 $\left( r, \ \theta \right)$ 点都代表一条经过 $\left( x_{0}, \ y_{0} \right)$ 的直线

  - 如果两族直线 $r_{\theta_{1}}, \ r_{\theta_{2}}$ 在 $r - \theta$ 平面内相交于点 $\left( r_{0}, \ \theta_{0} \right)$，说明该点代表的直线同时经过 $\left( x_{1}, \ y_{1} \right), \ \left( x_{2}, \ y_{2} \right)$，即两点在同一条直线上

  - 在 $r - \theta$ 平面内，相交于某一点的曲线越多，该点代表的直线将由更多的点组成。因此，可以通过计算所有 $\left( r, \ \theta \right)$ 点处相交曲线的数量来寻找图像中的直线

<center>
<img src="images/hough_line_intersect.png"/>
</center>

### 算法流程

- 对图像进行边缘检测，并二值化处理

- 将所有非零像素点变换到霍夫空间并累加到霍夫表中，统计累加值

- 记录累加值大于阈值并且为邻域最大值的点

- 将上述 $\left( r, \ \theta \right)$ 点转换为直线，并进一步处理

## 概率霍夫线变换

### 算法流程

- 从边缘特征图中随机选取一个非零像素点：

  - 如果该点已被标定为某一条直线上的点，则继续随机选择非零像素点

- 将该点变换到霍夫空间，并累加到霍夫表中

- 选取霍夫空间表中值最大的点：

  - 如果该点小于阈值，继续从边缘特征图中随机选取非零像素点

  - 如果该点大于阈值，从该点出发沿着直线方向，寻找线段端点：

    - 如果线段长度超过某个阈值，则将该线段输出

## $\mathrm{Python}$ 实现

### 标准霍夫线变换

```python
edge = cv2.Canny(image, threshold1, threshold2)
lines = cv2.HoughLines(edge, rho=1, theta=np.pi/180, threshold=100)
for line in lines:
  rho, theta = line[0]
  process(rho, theta)
```

### 概率霍夫线变换

```python
edge = cv2.Canny(image, threshold1, threshold2)
lines = cv2.HoughLinesP(edge, rho=1, theta=np.pi/180, threshold=100, maxLineGap=3)

for line in lines:
  x1, y1, x2, y2 = line[0]
  process((x1, y1), (x2, y2))
```