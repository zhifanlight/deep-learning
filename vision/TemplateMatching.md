<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 模版匹配

## 原理分析

- 给定模版 \\( T \\)、待搜索图像 \\( I \\)，在 \\( I \\) 中搜索与 \\( T \\) 最相似的区域；可用于简单的目标追踪

- 遍历 \\( I \\) 中的每个位置，以位置为左上角、与 \\( T \\) 同尺寸的图像块记为 \\( C \\)，该位置得分为 \\( C \\) 和 \\( T \\) 相似度得分

## 常用方法

### 平方差匹配

- 得分为 \\( C \\) 和 \\( T \\) 所有像素点像素差的平方和

- 值越小，相似度越高
	
	$$ R(x, y) = \sum\_{{x}', {y}'} [T({x}', {y}') - I(x + {x}', y + {y}')] ^{2} $$

### 归一化平方差匹配

- 相似度为 \\( C \\) 和 \\( T \\) 所有像素点像素差的平方和 除以  \\( C \\) 和 \\( T \\) 各自像素值平方和乘积的平方根

- 值越小，相似度越高

	$$ S(x, y) = \frac {\sum\_{{x}', {y}'} [T({x}', {y}') - I(x + {x}', y + {y}')] ^{2}} {\sqrt {\sum\_{{x}', {y}'} T({x}', {y}') ^{2} \cdot \sum\_{{x}', {y}'} I(x + {x}', y + {y}') ^{2} }} $$
	
### 相关性匹配

- 得分为 \\( C \\) 和 \\( T \\) 所有像素点像素值乘积和

- 值越大，相似度越高

	$$ R(x, y) = \sum\_{{x}', {y}'} [T({x}', {y}') \cdot I(x + {x}', y + {y}')] $$

### 归一化相关性匹配

- 得分为 \\( C \\) 和 \\( T \\) 所有像素点像素值乘积和 除以  \\( C \\) 和 \\( T \\) 各自像素值平方和乘积的平方根

- 值越大，相似度越高

	$$ S(x, y) = \frac {\sum\_{{x}', {y}'} [T({x}', {y}') \cdot I(x + {x}', y + {y}')] } {\sqrt {\sum\_{{x}', {y}'} T({x}', {y}') ^{2} \cdot \sum\_{{x}', {y}'} I(x + {x}', y + {y}') ^{2} }} $$

### 相关系数匹配

- 得分为 \\( C \\) 和 \\( T \\) 所有像素点像素值减去各自均值后的乘积和

- 值越大，相似度越高

	$$ R(x, y) = \sum\_{{x}', {y}'} [{T}' ({x}', {y}') \cdot {I}' (x + {x}', y + {y}')] $$

- 其中，

$$ \\left\\{ \begin{matrix} {T}' ({x}', {y}') =  T({x}', {y}') - \bar {T} \\\\ {I}' (x + {x}', y + {y}') =  I(x + {x}', y + {y}') - \bar {C} \end{matrix} \\right\.$$

### 归一化相关系数匹配

- 得分为 \\( C \\) 和 \\( T \\) 所有像素点像素值减去各自均值后的乘积和 除以  \\( C \\) 和 \\( T \\) 各自像素值减去各自均值后平方和乘积的平方根

- 值越大，相似度越高

	$$ S(x, y) = \frac {\sum\_{{x}', {y}'} [{T}' ({x}', {y}') \cdot {I}' (x + {x}', y + {y}')] } {\sqrt {\sum\_{{x}', {y}'} {T}' ({x}', {y}') ^{2} \cdot \sum\_{{x}', {y}'} {I}' (x + {x}', y + {y}') ^{2} }} $$

## Python 实现

### 平方差匹配

```
response = cv2.matchTemplate(image, tmpl, cv2.TM_SQDIFF)
```

### 归一化平方差匹配

```
response = cv2.matchTemplate(image, tmpl, cv2.TM_SQDIFF_NORMED)
```

### 相关性匹配

```
response = cv2.matchTemplate(image, tmpl, cv2.TM_CCORR)
```
	
### 归一化相关性匹配

```
response = cv2.matchTemplate(image, tmpl, cv2.TM_CCORR_NORMED)
```
	
### 相关系数匹配

```
response = cv2.matchTemplate(image, tmpl, cv2.TM_CCOEFF)
```
	
### 归一化相关系数匹配

```
response = cv2.matchTemplate(image, tmpl, cv2.TM_CCOEFF_NORMED)
```