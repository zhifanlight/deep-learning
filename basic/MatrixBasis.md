<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 矩阵基础

## 方阵的迹

- 主对角线上元素之和

### 迹的性质

#### 性质 1：转置不变

$$ tr(A)=tr(A^{T}) $$

#### 性质 2：循环不变

$$ tr(AB) = tr(BA), \quad tr(ABC) = tr(BCA) = tr(CAB), \quad ... $$

- 设 \\(A\\) 为 \\(M \times N\\) 矩阵，\\(B\\) 为 \\(N \times M\\) 矩阵：

	$$ tr(AB) = \sum\_{i=1}^{M} (AB)\_{ii} = \sum\_{i=1}^{M}\sum\_{j=1}^{N} a\_{ij}b\_{ji} = \sum\_{j=1}^{N}\sum\_{i=1}^{M} b\_{ji}a\_{ij} = \sum\_{j=1}^{N} (BA)\_{jj} = tr(BA) $$

- 由 \\(tr(AB) = tr(BA)\\) 可得：\\(tr(ABC) = tr(BCA) = tr(CAB)\\)

#### 性质 3：线性可加

$$ tr(\alpha \cdot A + \beta \cdot B) = \alpha \cdot tr(A) + \beta \cdot tr(B) $$

- 其中，\\(\alpha, \beta\\) 均为标量，\\(A, B\\) 为同维度方阵

## 矩阵求导

- 若存在矩阵 \\(H\\) 使下式成立，则 \\(\nabla\_{X}f(X) = H\\)，

	$$ \lim\_{t \rightarrow 0} \frac{f(X + t \cdot W) - f(X)}{t} = tr(W^{T}H) $$
	
	- 其中 \\(W\\) 表示与 \\(X\\) 维度相同的任意矩阵

- 由于机器学习的输出通常为标量，因此可以假设 \\(f(X) = tr(X)\\)，然后进行求导