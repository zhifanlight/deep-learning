<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 范数

&nbsp;

## 向量范数

### 0-范数

- 向量中非零元素个数：

$$
||x||\_{0} = \sum\_{i=1}^{N}I(x\_{i} \neq 0)
$$

### 1-范数

- 向量元素绝对值之和：

$$
||x||\_{1} = \sum\_{i=1}^{N}|x\_{i}|
$$

### 2-范数（Euclid 范数）

- 向量元素平方和的平方根：

$$
||x||\_{2} = \sqrt{\sum\_{i=1}^{N} x\_{i}^{2}}
$$

### \\(\infty\\)-范数（无穷范数）

- 向量元素绝对值中最大值：

$$
||x||\_{\infty} = max|x\_{i}|
$$

### \\(-\infty\\)-范数

- 向量元素绝对值中最小值：

$$
||x||\_{-\infty} = min|x\_{i}|
$$

### p-范数

- 向量元素绝对值 p 次方和的 p 次方根

$$
||x||\_{p} = \left(\sum\_{i=1}^{N} |x\_{i}|^{p}\right)^{\frac{1}{p}}
$$