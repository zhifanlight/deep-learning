<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 高斯分布

## 概率分布

### 单变量

$$ p(x) = \frac{1}{\sqrt{2 \pi} \sigma} exp \left( -\frac{(x-\mu)^{2}}{2 \sigma^{2}} \right) $$

- 其中 \\(\mu, \ \sigma^{2}\\) 分别是均值和方差

### 多元

$$ p(x) = \frac{1}{(2 \pi)^{\frac{n}{2}} |\Sigma|^{\frac{1}{2}}} exp \left( -\frac{1}{2} (x-\mu)^{T} \Sigma^{-1} (x-\mu) \right) $$

- 其中 \\(\mu, \ \Sigma\\) 分别是均值向量和协方差矩阵

## 数学推导

### \\(E(x) = \mu\\)

- 由概率之和 \\( \int p(x) \ dx = 1\\)：

	$$ \int exp \left( -\frac{(x-\mu)^{2}}{2 \sigma^{2}} \right) \ dx = \sqrt{2 \pi} \sigma $$

- 同时对 \\(\mu\\) 求导：

	$$ \int exp \left( -\frac{(x-\mu)^{2}}{2 \sigma^{2}} \right) \cdot \left( \frac{2 \cdot (x-\mu)}{2 \sigma^{2}} \right) \ dx = 0 $$

- 对上式进行化简：

	$$ \int exp \left( -\frac{(x-\mu)^{2}}{2 \sigma^{2}} \right) \cdot x \ dx = \mu \int exp \left( -\frac{(x-\mu)^{2}}{2 \sigma^{2}} \right) \ dx = \mu \sqrt{2 \pi} \sigma $$

- 由定义 \\(E(x) = \int p(x) \cdot x \ dx\\)：

	$$ E(x) = \frac{1}{\sqrt{2 \pi} \sigma} \int exp \left( -\frac{(x-\mu)^{2}}{2 \sigma^{2}} \right) \cdot x \ dx $$

- 将化简结果代入上式：

	$$ E(x) = \mu $$

### \\(Var(x) = \sigma^{2}\\)

- 由概率之和 \\( \int p(x) \ dx = 1\\)：

	$$ \int exp \left( -\frac{(x-\mu)^{2}}{2 \sigma^{2}} \right) \ dx = \sqrt{2 \pi} \sigma $$

- 用 \\(\sigma^{2}=t\\) 对上式进行替换：

	$$ \int exp \left( -\frac{(x-\mu)^{2}}{2t} \right) \ dx = \sqrt{2 \pi t} $$

- 同时对 \\(t\\) 求导：

	$$ \int exp \left( -\frac{(x-\mu)^{2}}{2t} \right) \cdot \left( \frac{(x-\mu)^{2}}{2t^{2}} \right) \ dx = \sqrt{2 \pi} \cdot \frac{1}{2} \cdot t^{-\frac{1}{2}} $$

- 同时乘上 \\(2 t^{2}\\)：

	$$ \int exp \left( -\frac{(x-\mu)^{2}}{2t} \right) \cdot (x-\mu)^{2} \ dx = \sqrt{2 \pi t} \cdot t $$

- 将 \\(t=\sigma^{2}\\) 代回上式：

	$$ \int \frac{1}{\sqrt{2 \pi} \sigma} exp \left( -\frac{(x-\mu)^{2}}{2 \sigma^{2}} \right) \cdot (x-\mu)^{2} \ dx = \sigma^{2} $$

- 由定义 \\(Var(x) = E \left( (x-\mu)^{2} \right)\\)：

	$$ Var(x) = \int p(x) \cdot (x-\mu)^{2} \ dx $$

- 将 \\(p(x)\\) 代入上式：

	$$ Var(x) = \sigma^{2} $$