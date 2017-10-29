<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 二值图像连通区域分析

&nbsp;

## 原理分析

- 将二值图像中不同的连通区域分离开

- 常见方法

	- Two Pass（两遍扫描）

		- 按行或列遍历二值图像的所有非零元素：

			- 如果当前元素邻域内的所有元素都为 0，为当前元素赋予一个新标签；否则，将当前元素标签赋值为邻域内的最小非零标签

			- 如果当前元素邻域内有多种标签，称这些标签两两之间存在相等关系

		- 将所有存在相等关系的标签对合并成一个大的标签（比如通过深搜）

		- 再一次遍历图像，将原始标签更新为合并后的新标签

	- Seed Filling（种子填充）

		- 遍历二值图像的所有非零元素：

			- 从当前元素开始进行深搜，直到找出所有与当前元素相连的元素；在搜索的同时，将当前元素的标签赋予相应位置
		
		- 【虽然只遍历一遍，但使用 Python 实现时，运行时间是 Two Pass 的两倍】

## 连通区域重心

- \\(\bar{x} = \frac{\sum\_{i}\sum\_{j} {x\_{i} \cdot p\_{i,j}}}{A}\\)，\\(\bar{y} = \frac{\sum\_{i}\sum\_{j} {y\_{j} \cdot p\_{i,j}}}{A}\\)

- \\(A\\) 是连通区域的面积，\\(x\_{i}, y\_{j}, p_{i,j}\\) 分别是 \\((i, j)\\) 位置的行坐标、列坐标、灰度值

## Python 实现

```
# cnt: connected component count
# res: label for every pixel, range from 0 to cnt - 1

cnt, res = cv2.connectedComponents(image)
```