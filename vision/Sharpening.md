<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 图像锐化

## 原理介绍

- 常用 Laplacian 边缘特征进行锐化

- 原图像为 \\(R\\)，边缘特征为 \\(E\\)，锐化后得到 \\(S\\)：

	- 若 \\( G = \\left[ \\begin{matrix} 0 & 1 & 0 \\\\ 1 & -4 & 1 \\\\ 0 & 1 & 0 \\end{matrix} \\right] \\)，则 \\(S = abs(R - E)\\)

	- 若 \\( G = \\left[ \\begin{matrix} 0 & -1 & 0 \\\\ -1 & 4 & -1 \\\\ 0 & -1 & 0 \\end{matrix} \\right] \\)，则 \\(S = abs(R + E)\\)

## Python 实现

```
edge = cv2.Laplacian(image, cv2.CV_16S)
response = cv2.convertScaleAbs(image - edge)
```

或者

```
kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
response = cv2.filter2D(image, cv2.CV_16S, kernel)
response = cv2.convertScaleAbs(response)
```