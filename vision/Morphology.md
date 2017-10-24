<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 形态学运算

&nbsp;

## 腐蚀

- 将当前位置灰度值更新为邻域内最小值

- 操作完成后，图像高灰度值区域变小

## 膨胀

- 将当前位置灰度值更新为邻域内最大值

- 操作完成后，图像高灰度值区域变大


## 开运算

- 先腐蚀，再膨胀

- 消除小物体，从连接较细处分离物体，平滑大物体边界

## 闭运算

- 先膨胀，再腐蚀

- 填充离散小空洞，连接相邻物体

&nbsp;

## Python 实现

### 创建卷积核

```
# rectangle with size (width, height)
kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (width, height))
```

```
# ellipse in box (width, height)
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (width, height))
```

### 腐蚀（分别处理各通道）

```
response = cv2.erode(frame, kernel)
```

### 膨胀（分别处理各通道）

```
response = cv2.dilate(frame, kernel)
```

### 开运算

```
response = cv2.erode(frame, kernel)
response = cv2.dilate(response, kernel)
```

### 闭运算

```
response = cv2.dilate(frame, kernel)
response = cv2.erode(response, kernel)
```