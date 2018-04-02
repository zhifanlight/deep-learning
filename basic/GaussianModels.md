<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 高斯模型

## 概率分布

### 一维

$$ p(x) = \frac{1}{\sqrt{2 \pi} \sigma} exp \left( -\frac{(x-\mu)^{2}}{2 \sigma^{2}} \right) $$

- 其中 \\(\mu, \ \sigma^{2}\\) 分别是均值和方差

### 多维

$$ p(x) = \frac{1}{(2 \pi)^{\frac{n}{2}} |\Sigma|^{\frac{1}{2}}} exp \left( -\frac{1}{2} (x-\mu)^{T} \Sigma^{-1} (x-\mu) \right) $$

- 其中 \\(\mu, \ \Sigma\\) 分别是均值向量和协方差矩阵

### 数学推导

#### \\(E(x) = \mu\\)

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

#### \\(Var(x) = \sigma^{2}\\)

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

## 高斯混合模型

### 背景介绍

- 假设样本集来自 \\(k\\) 个高斯分布，即 \\(x\_{i} \sim N(\mu\_{j}, \sigma\_{j}^{2})\\)，求每个样本采样的高斯分布

- 即 \\(x\_{i}\\) 已知但 \\(z\_{i}\\) 未知，求 \\(P(z\_{i}|x\_{i} \ ; \theta)\\)，其中 \\(z\_{i} \sim M(\phi)\\)

### 数学推导

- 对数似然函数如下：

	$$
	\begin{align\*}
	L(\phi, \mu, \sigma^{2}) &= \sum\_{i=1}^{m} log \ P(x\_{i} \ ; \phi, \mu, \sigma^{2}) \newline
	&= \sum\_{i=1}^{m} log \left( \sum\_{j=1}^{k} P(z\_{i}=j \ ; \phi) \cdot P(x\_{i}|z\_{i}=j \ ; \mu, \sigma^{2}) \right) \newline
	&= \sum\_{i=1}^{m} log \left( \sum\_{j=1}^{k} Q\_{i}(z\_{i}=j) \cdot \frac{P(z\_{i}=j \ ; \phi) \cdot P(x\_{i}|z\_{i}=j \ ; \mu, \sigma^{2})}{Q\_{i}(z\_{i}=j)} \right) \newline
	&\geq \sum\_{i=1}^{m} \sum\_{j=1}^{k} Q\_{i}(z\_{i}=j) \cdot log \left( \frac{P(z\_{i}=j \ ; \phi) \cdot P(x\_{i}|z\_{i}=j \ ; \mu, \sigma^{2})}{Q\_{i}(z\_{i}=j)} \right)
	\end{align\*}
	$$

- 定义 \\(w\_{j}^{i}\\)：

	$$ w\_{j}^{i} = P(z\_{i}=j|x\_{i} \ ; \phi, \mu, \sigma^{2}) = Q\_{i}(z\_{i}=j) $$

	- 进一步推导可得：

		$$ P(z\_{i}=j|x\_{i}) = \frac{P(z\_{i}=j, x\_{i})}{P(x\_{i})} = \frac{P(z\_{i}=j) \cdot P(x\_{i}|z\_{i}=j)}{\sum\_{l=1}^{k} P(z\_{i}=l) \cdot P(x\_{i}|z\_{i}=l)} $$

	- 由分布形式 \\(z\_{i} \sim M(\phi), \ x\_{i} \sim N(\mu\_{j}, \sigma\_{j}^{2})\\)：

		$$ w\_{j}^{i} = \frac{\phi\_{j} \cdot \sigma\_{j}^{-1} \cdot exp \left( \frac{(x\_{i}-\mu\_{j})^{2}}{\sigma\_{j}^{2}} \right)}{\sum\_{l=1}^{k} \phi\_{l} \cdot \sigma\_{l}^{-1} exp \left( \frac{(x\_{i}-\mu\_{l})^{2}}{\sigma\_{l}^{2}} \right) } $$

- 代入对数似然函数：

	$$ L(\phi, \mu, \Sigma) \geq \sum\_{i=1}^{m} \sum\_{j=1}^{k} w\_{j}^{i} \left( log \ \phi\_{j} + \left( log \ \frac{1}{\sqrt{2 \pi \sigma\_{j}^{2}}} - \frac{(x\_{i}-\mu\_{j})^{2}}{2 \sigma\_{j}^{2}} \right) - log \ w\_{j}^{i} \right) $$

#### 固定 \\(\mu, \sigma^{2}\\) 对 \\(\phi\\) 求导

- 忽略常数项，极大似然函数如下：

	$$ L(\phi) = \sum\_{i=1}^{m} \sum\_{j=1}^{k} w\_{j}^{i} \cdot log \ \phi\_{j} + C $$

- 约束条件如下：

	$$ \sum\_{j=1}^{k} \phi\_{j} = 1 $$
		
	$$ \phi\_{j} \geq 0 \quad j = 1,2,\cdots,k $$

- 由拉格朗日乘子法：

	$$ L(\phi, \beta) = -\sum\_{i=1}^{m} \sum\_{j=1}^{k} w\_{j}^{i} \cdot log \ \phi\_{j} + \beta \left( \sum\_{j=1}^{k} \phi\_{j} - 1 \right) + C $$

	- 临时忽略约束条件 \\(\phi\_{j} \geq 0\\)

- 对 \\(\phi\_{j}\\) 求导并令导数为 \\(0\\)：

	$$ \phi\_{j} = \frac{1}{\beta} \sum\_{i=1}^{m} w\_{j}^{i} $$

- 由概率之和 \\(\sum\_{j=1}^{k} \phi\_{j}=1, \ \sum\_{j=1}^{k} w\_{j}^{i}=1\\)：

	$$ 1 = \sum\_{j=1}^{k} \frac{1}{\beta} \sum\_{i=1}^{m} w\_{j}^{i} = \frac{1}{\beta} \sum\_{i=1}^{m} \sum\_{j=1}^{k} w\_{j}^{i} = \frac{m}{\beta} $$

- 将化简结果 \\(\beta = m\\) 代入求导结果：

	$$ \phi\_{j} = \frac{1}{m} \sum\_{i=1}^{m} w\_{j}^{i} $$

	- 满足被忽略的约束条件 \\(\phi\_{j} \geq 0\\)

#### 固定 \\(\phi, \sigma^{2}\\) 对 \\(\mu\\) 求导

- 忽略常数项，极大似然函数如下：

	$$ L(\mu) = -\sum\_{i=1}^{m} \sum\_{j=1}^{k} w\_{j}^{i} \cdot \frac{(x\_{i}-\mu\_{j})^{2}}{2 \sigma\_{j}^{2}} + C $$

- 对 \\(\mu\_{j}\\) 求导并令导数为 \\(0\\)：

	$$ \mu\_{j} = \frac{\sum\_{i=1}^{m} w\_{j}^{i} \cdot x\_{i}}{\sum\_{i=1}^{m} x\_{i}} $$

#### 固定 \\(\phi, \mu\\) 对 \\(t=\sigma^{2}\\) 求导

- 忽略常数项，极大似然函数如下：

	$$ L(t) = \sum\_{i=1}^{m} \sum\_{j=1}^{k} w\_{j}^{i} \cdot \left( log \frac{1}{\sqrt{2 \pi t\_{j}}} - \frac{(x\_{i}-\mu\_{j})^{2}}{2 t\_{j}} \right) + C $$

- 对 \\(t\_{j}\\) 求导并令导数为 \\(0\\)：

	$$ t\_{j} = \frac{\sum\_{i=1}^{m} w\_{j}^{i} \cdot (x\_{i}-\mu\_{j})^{2}}{\sum\_{i=1}^{m} w\_{j}^{i}} $$

- 将 \\(t=\sigma^{2}\\) 代回上式：

	$$ \sigma\_{j}^{2} = \frac{\sum\_{i=1}^{m} w\_{j}^{i} \cdot (x\_{i}-\mu\_{j})^{2}}{\sum\_{i=1}^{m} w\_{j}^{i}} $$

### 一般形式

#### E 步

$$ w\_{j}^{i} = \frac{\phi\_{j} \cdot \sigma\_{j}^{-1} \cdot exp \left( \frac{(x\_{i}-\mu\_{j})^{2}}{\sigma\_{j}^{2}} \right)}{\sum\_{l=1}^{k} \phi\_{l} \cdot \sigma\_{l}^{-1} exp \left( \frac{(x\_{i}-\mu\_{l})^{2}}{\sigma\_{l}^{2}} \right) } $$

#### M 步

$$ \phi\_{j} = \frac{1}{m} \sum\_{i=1}^{m} w\_{j}^{i} $$

$$ \mu\_{j} = \frac{\sum\_{i=1}^{m} w\_{j}^{i} \cdot x\_{i}}{\sum\_{i=1}^{m} x\_{i}} $$

$$ \sigma\_{j}^{2} = \frac{\sum\_{i=1}^{m} w\_{j}^{i} \cdot (x\_{i}-\mu\_{j})^{2}}{\sum\_{i=1}^{m} w\_{j}^{i}} $$