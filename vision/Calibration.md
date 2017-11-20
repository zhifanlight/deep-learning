<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 相机校准

## 径向畸变

- 远离中心的直线变成曲线

- 畸变修正

	$$ \\left\\{ \\begin{matrix} {x}' = x \cdot (1 + k _{1} \cdot r ^2 + k _{2} \cdot r ^4 + k _{3} \cdot r ^6) \\\\ {y}' = y \cdot (1 + k _{1} \cdot r ^2 + k _{2} \cdot r ^4 + k _{3} \cdot r ^6) \\\\ r ^2 = x ^2 + y ^2 \\end{matrix} \\right. $$

## 切向畸变

- 由于镜头不平行，导致图像部分区域离镜头更近

- 畸变修正

	$$ \\left\\{ \\begin{matrix} {x}' = x + [2 \cdot p _{1} \cdot x \cdot y + p _{2} \cdot (r ^2 + 2 \cdot x ^2)] \\\\ {y}' = y + [2 \cdot p _{2} \cdot x \cdot y + p _{1} \cdot (r ^2 + 2 \cdot y ^2)] \\\\ r ^2 = x ^2 + y ^2 \\end{matrix} \\right. $$

## 综合情况

- 畸变修正

	$$ \\left\\{ \\begin{matrix} {x}' = x \cdot (1 + k _{1} \cdot r ^2 + k _{2} \cdot r ^4 + k _{3} \cdot r ^6) + [2 \cdot p _{1} \cdot x \cdot y + p _{2} \cdot (r ^2 + 2 \cdot x ^2)] \\\\ {y}' = y \cdot (1 + k _{1} \cdot r ^2 + k _{2} \cdot r ^4 + k _{3} \cdot r ^6) + [2 \cdot p _{2} \cdot x \cdot y + p _{1} \cdot (r ^2 + 2 \cdot y ^2)]  \\end{matrix} \\right. $$

- 畸变向量 \\( distortion = [k \_{1}, k \_{2}, p \_{1}, p \_{2}, k \_{3}] \\)

- 相机参数 \\( matrix=\\left[ \\begin{matrix} f \_{x} & 0 & c \_{x} \\\\ 0 & f \_{y} & c \_{y} \\\\ 0 & 0 & 1 \\end{matrix} \\right] \\)，其中，\\( \\left( \\begin{matrix}f \_{x} & f \_{y} \\end{matrix} \\right) \\)是相机焦距，\\( \\left( \\begin{matrix}c \_{x} & c \_{y} \\end{matrix} \\right) \\)是相机光学中心

## Python 实现

### 调优矩阵

```
new_matrix, _ = cv2.getOptimalNewCameraMatrix(matrix, distortion, size, 1, size)
```

### 图像校准

- undistort

```
new_image = cv2.undistort(image, matrix, distortion, None, new_matrix)
```

- remap

```
map_x, map_y = cv2.initUndistortRectifyMap(matrix, distortion, None, new_matrix, (width, height), cv2.CV_32FC1)
new_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
```

### 坐标校准

```
points = [point1, point2, ..., pointN]
points = numpy.array(points, numpy.float32).reshape(1, len(points), -1)
points = cv2.undistortPoints(points, matrix, distortion)[0]
fx, fy = new_matrix[0][0], new_matrix[1][1]
cx, cy = new_matrix[0][2], new_matrix[1][2]
for idx in range(len(points)):
	points[idx][0] = int(round(points[idx][0] * fx + cx))
	points[idx][1] = int(round(points[idx][1] * fy + cy))
```

### 坐标畸变

```
def distort(point, matrix, new_matrix, distortion):
	fx, fy = new_matrix[0][0], new_matrix[1][1]
	cx, cy = new_matrix[0][2], new_matrix[1][2]
	x, y = (point[0] - cx) / fx, (point[1] - cy) / fy
	
	k1, k2, p1, p2, k3 = distortion[0]
	r2 = x ** 2 + y ** 2
	r4 = r2 * r2
	r6 = r2 * r2 * r2
		
	xx = x * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
	yy = y * (1 + k1 * r2 + k2 * r4 + k3 * r6) + 2 * p2 * x * y + p1 * (r2 + 2 * y * y)
	
	fx, fy = matrix[0][0], matrix[1][1]
	cx, cy = matrix[0][2], matrix[1][2]
	return int(round(xx * fx + cx)), int(round(yy * fy + cy))
```