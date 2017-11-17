<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 重映射

## 原理分析

- 在原图像和新图像间建立点映射关系： \\( g(x, y) = f(h(x, y)) \\)

- 其中，\\( f \\) 是原图像，\\( g \\) 是新图像，而 \\( h \\) 是作用于 \\( (x, y) \\) 的映射方法

- 对于水平翻转：\\( g(x, y) = f(cols - x, y) \\)

## Python 实现

```
for row in range(height):
	for col in range(width):
		map_x[row, col] = width - 1 - col
		map_y[row, col] = row
new_image = cv2.remap(image, map_x, map_y, cv2.INTER_LINEAR)
```