<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 矩阵基础

## 正定阵

- 一个实对称矩阵 \\(M\\) 是正定的，当且仅当对所有非零实向量 \\(z\\)，都有 \\(z^{T}Mz > 0\\)

- 正定阵的特征值都是正数

## 半正定阵

- 一个实对称矩阵 \\(M\\) 是半正定的，当且仅当对所有非零实向量 \\(z\\)，都有 \\(z^{T}Mz \geq 0\\)

- 半正定阵的特征值都非负

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

## 方阵特征值、特征向量

- 方阵 \\(A\\) 的特征值 \\(\lambda\\)、特征向量 \\(v\\) 满足以下关系：

	$$ Av = \lambda v \quad \Leftrightarrow \quad (A - \lambda I)v = 0 $$

- 令 \\(det(A - \lambda I) = 0\\) 可计算特征值，将特征值代入上式可计算特征向量

- 迹等于特征值之和，行列式等于特征值之积：

	- 假设 \\( A=\\left[ \begin{matrix} a & b \\\\ c & d \end{matrix} \\right], v = \\left[ \begin{matrix} m \\\\ n \end{matrix} \\right] \\)，代入上式并化简可得：

		$$ \lambda^{2} - (a + d) \lambda + (ad - bc) = 0 $$

	- 求解上述一元二次方程可得：

		$$ \\left\\{ \begin{matrix} \lambda\_{1} + \lambda\_{2} = a + d = tr(A) \\\\ \lambda\_{1} \cdot \lambda\_{2} = ad - bc = det(A) \end{matrix} \\right. $$

## 矩阵求导

- 若存在矩阵 \\(H\\) 使下式成立，则 \\(\nabla\_{X}f(X) = H\\)，

	$$ \lim\_{t \rightarrow 0} \frac{f(X + t \cdot W) - f(X)}{t} = tr(W^{T}H) $$
	
	- 其中 \\(W\\) 表示与 \\(X\\) 维度相同的任意矩阵

- 由于机器学习的输出通常为标量，因此可以假设 \\(f(X) = tr(X)\\)，然后进行求导