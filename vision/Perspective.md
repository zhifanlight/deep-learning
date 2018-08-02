<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 透视变换

## 刚性变换

- 变换前后，图像中任意两点的距离不变，即图像形状不变

- 平移、旋转、翻转

## 仿射变换

- 变换前后，平行线依然平行，共线点依然共线。仿射变换前后的图像均为平行四边形

- 刚性变换、缩放、切变

## 透视变换

- 将一个平面内的图像先变换到三维空间，再投影到另一个平面内

- 任意形状的变换

## 原理分析

- 刚性变换是仿射变换的子集，仿射变换是透视变换的子集

- 其中，刚性变换、仿射变换的变换矩阵为 2 x 3 矩阵 \\( A \\)；透视变换的变换矩阵为 3 x 3 矩阵 \\( P \\) 

- \\(A = \\left[ \\begin{matrix} a & b & c \\\\ d & e & f \\end{matrix} \\right] \\)，变换公式为\\( \\left[ \\begin{matrix} x \\\\ y \\end{matrix} \\right] = A \\left[ \\begin{matrix} u \\\\ v \\\\ 1 \\end{matrix} \\right] \\) ，最终坐标为\\( \\left[ \\begin{matrix} x \\\\ y \\end{matrix} \\right] \\)

-  \\(P = \\left[ \\begin{matrix} a & b & c \\\\ d & e & f \\\\ g & h & i \\end{matrix} \\right] \\)，变换公式为\\( \\left[ \\begin{matrix} {x}' \\\\ {y}' \\\\ {z}' \\end{matrix} \\right] = P \\left[ \\begin{matrix} u \\\\ v \\\\ w \\end{matrix} \\right] \\) ，最终坐标为\\( \\left[ \\begin{matrix} {x}'/{z}' \\\\ {y}'/{z}' \\end{matrix} \\right] \\)

- 仿射矩阵 \\( A \\) 中有 6 个未知数，三组对应顶点即可求解

	- 从几何角度，仿射变换前后都是平行四边形，只需要三个顶点即可确定平行四边形的位置和形状

- 透视矩阵 \\( P \\) 中有 8 个未知数（将 \\( P \\) 缩放至 i ＝ 1 ），需要四组对应顶点进行求解

	- 从几何角度，透视变换前后只是普通的四边形，需要四个顶点才能确定四边形的位置和形状

- 透视变化通常用于不同坐标系间的坐标变换

## Python 实现

### 仿射变换

```
matrix = cv2.getAffineTransform(from, to)
	
trans = cv2.warpAffine(image, matrix, new_size)
```
### 透视变换

```
matrix = cv2.getPespectiveTransform(from, to)
	
trans = cv2.warpPerspective(image, matrix, new_size)
```