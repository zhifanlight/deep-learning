# 高斯模型

## 概率分布

### 一维

$$
p \left( x \right) = \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( -\frac{\left( x - \mu \right)^{2}}{2 \sigma^{2}} \right)
$$

- 其中 $\mu, \ \sigma^{2}$ 分别是均值和方差

### 多维

$$
p \left( x \right) = \frac{1}{\left( 2 \pi \right)^{\frac{n}{2}} |\Sigma|^{\frac{1}{2}}} \exp \left( -\frac{1}{2} \left( x - \mu \right)^{T} \Sigma^{-1} \left( x - \mu \right) \right)
$$

- 其中 $\mu, \ \Sigma$ 分别是均值向量和协方差矩阵

### 数学推导

#### $\mathbb{E} \left( x \right) = \mu$

- 由概率之和 $ \int p \left( x \right) \mathrm{d} x = 1$：

  $$
  \int \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( -\frac{ \left( x - \mu \right)^{2}}{2 \sigma^{2}} \right) \mathrm{d} x = 1
  $$

- 同时对 $\mu$ 求导：

  $$
  \int \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( -\frac{\left( x - \mu \right)^{2}}{2 \sigma^{2}} \right) \cdot \left( \frac{2 \cdot \left( x - \mu \right)}{2 \sigma^{2}} \right) \mathrm{d} x = 0
  $$

- 对上式进行化简：

  $$
  \int \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( -\frac{ \left( x - \mu \right)^{2}}{2 \sigma^{2}} \right) \cdot x \mathrm{d} x = \mu \int \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( -\frac{\left( x - \mu \right)^{2}}{2 \sigma^{2}} \right) \mathrm{d}x = \mu
  $$

- 由定义 $\mathbb{E} \left( x \right) = \int p \left( x \right) \cdot x \mathrm{d} x$：

  $$
  \mathbb{E} \left( x \right) = \mu
  $$

#### $\mathrm{Var} \left( x \right) = \sigma^{2}$

- 由概率之和 $ \int p \left( x \right) \mathrm{d} x = 1$：

  $$
  \int \exp \left( -\frac{ \left( x -\mu \right)^{2}}{2 \sigma^{2}} \right) \mathrm{d} x = \sqrt{2 \pi} \sigma
  $$

- 用 $\sigma^{2} = t$ 对上式进行替换：

  $$
  \int \exp \left( -\frac{ \left( x - \mu \right)^{2}}{2t} \right) \mathrm{d} x = \sqrt{2 \pi t}
  $$

- 同时对 $t$ 求导：

  $$
  \int \exp \left( -\frac{ \left( x - \mu \right)^{2}}{2t} \right) \cdot \left( \frac{ \left( x - \mu \right)^{2}}{2t^{2}} \right) \mathrm{d} x = \sqrt{2 \pi} \cdot \frac{1}{2} \cdot t^{-\frac{1}{2}}
  $$

- 同时乘上 $2 t^{2}$：

  $$
  \int \exp \left( -\frac{ \left( x - \mu \right)^{2}}{2t} \right) \cdot \left( x - \mu \right)^{2} \mathrm{d} x = \sqrt{2 \pi t} \cdot t
  $$

- 将 $t = \sigma^{2}$ 代回上式：

  $$
  \int \frac{1}{\sqrt{2 \pi} \sigma} \exp \left( -\frac{\left( x - \mu \right)^{2}}{2 \sigma^{2}} \right) \cdot \left( x - \mu \right)^{2} \mathrm{d} x = \sigma^{2}
  $$

- 由定义 $\mathrm{Var} \left( x \right) = \mathbb{E} \left( \left( x - \mu \right)^{2} \right)$：

  $$
  \mathrm{Var} \left( x \right) = \int p \left( x \right) \cdot \left( x - \mu \right)^{2} \mathrm{d} x
  $$

- 将 $p \left( x \right)$ 代入上式：

  $$
  \mathrm{Var} \left( x \right) = \sigma^{2}
  $$

## 高斯混合模型

### 背景介绍

- 假设样本集来自 $k$ 个高斯分布，即 $x_{i} \sim N \left( \mu_{j}, \ \sigma_{j}^{2} \right)$，求每个样本采样的高斯分布

- 即 $x_{i}$ 已知但 $z_{i}$ 未知，求 $P \left( z_{i}|x_{i}; \ \theta \right)$，其中 $z_{i} \sim M \left( \phi \right)$

### 数学推导

- 对数似然函数如下：

  $$
  \begin{aligned}
  L \left( \phi, \ \mu, \ \sigma^{2} \right) &= \sum_{i = 1}^{m} \log P \left( x_{i}; \ \phi, \ \mu, \ \sigma^{2} \right) \newline
  &= \sum_{i = 1}^{m} \log \left( \sum_{j = 1}^{k} P \left( z_{i} = j; \phi \right) \cdot P \left( x_{i}|z_{i} = j; \ \mu, \ \sigma^{2} \right) \right) \newline
  &= \sum_{i = 1}^{m} \log \left( \sum_{j = 1}^{k} Q_{i} \left( z_{i} = j \right) \cdot \frac{P \left( z_{i} = j; \ \phi \right) \cdot P \left( x_{i}|z_{i} = j; \ \mu, \ \sigma^{2} \right)}{Q_{i} \left( z_{i} = j \right)} \right) \newline
  &\geq \sum_{i = 1}^{m} \sum_{j = 1}^{k} Q_{i} \left( z_{i} = j \right) \cdot \log \left( \frac{P \left( z_{i} = j; \ \phi \right) \cdot P \left( x_{i}|z_{i} = j; \ \mu, \ \sigma^{2} \right)}{Q_{i} \left( z_{i} = j \right)} \right)
  \end{aligned}
  $$

- 定义 $w_{j}^{i}$：

  $$
  w_{j}^{i} = P \left( z_{i} = j|x_{i}; \ \phi, \ \mu, \ \sigma^{2} \right) = Q_{i}\left( z_{i} = j \right)
  $$

  - 进一步推导可得：

    $$
    P \left( z_{i} = j|x_{i} \right) = \frac{P \left( z_{i} = j, \ x_{i} \right)}{P \left( x_{i} \right)} = \frac{P \left( z_{i} = j \right) \cdot P \left( x_{i}|z_{i} = j \right)}{\sum_{l = 1}^{k} P \left( z_{i} = l \right) \cdot P \left( x_{i}|z_{i} = l \right)}
    $$

  - 由分布形式 $z_{i} \sim M \left( \phi \right), \ x_{i} \sim N \left( \mu_{j}, \ \sigma_{j}^{2} \right)$：

    $$
    w_{j}^{i} = \frac{\phi_{j} \cdot \sigma_{j}^{-1} \cdot \exp \left( \frac{\left( x_{i} - \mu_{j} \right)^{2}}{\sigma_{j}^{2}} \right)}{\sum_{l = 1}^{k} \phi_{l} \cdot \sigma_{l}^{-1} \exp \left( \frac{\left( x_{i} - \mu_{l} \right)^{2}}{\sigma_{l}^{2}} \right) }
    $$

- 代入对数似然函数：

  $$
  L \left(\phi, \ \mu, \ \Sigma \right) \geq \sum_{i = 1}^{m} \sum_{j = 1}^{k} w_{j}^{i} \left( \log \phi_{j} + \left( \log \frac{1}{\sqrt{2 \pi \sigma_{j}^{2}}} - \frac{\left( x_{i} - \mu_{j} \right)^{2}}{2 \sigma_{j}^{2}} \right) - \log \ w_{j}^{i} \right)
  $$

#### 固定 $\mu, \ \sigma^{2}$ 对 $\phi$ 求导

- 忽略常数项，极大似然函数如下：

  $$
  L \left( \phi \right) = \sum_{i = 1}^{m} \sum_{j = 1}^{k} w_{j}^{i} \cdot \log \phi_{j} + C
  $$

- 约束条件如下：

  $$
  \sum_{j = 1}^{k} \phi_{j} = 1
  $$

  $$
  \phi_{j} \geq 0 \quad j = 1, \ 2, \ \cdots, \ k
  $$

- 由拉格朗日乘子法：

  $$
  L \left( \phi, \ \beta \right) = -\sum_{i = 1}^{m} \sum_{j = 1}^{k} w_{j}^{i} \cdot \log \phi_{j} + \beta \left( \sum_{j = 1}^{k} \phi_{j} - 1 \right) + C
  $$

  - 临时忽略约束条件 $\phi_{j} \geq 0$

- 对 $\phi_{j}$ 求导并令导数为 $0$：

  $$
  \phi_{j} = \frac{1}{\beta} \sum_{i = 1}^{m} w_{j}^{i}
  $$

- 由概率之和 $\sum_{j = 1}^{k} \phi_{j}=1, \ \sum_{j = 1}^{k} w_{j}^{i}=1$：

  $$
  1 = \sum_{j = 1}^{k} \frac{1}{\beta} \sum_{i = 1}^{m} w_{j}^{i} = \frac{1}{\beta} \sum_{i = 1}^{m} \sum_{j = 1}^{k} w_{j}^{i} = \frac{m}{\beta}
  $$

- 将化简结果 $\beta = m$ 代入求导结果：

  $$
  \phi_{j} = \frac{1}{m} \sum_{i = 1}^{m} w_{j}^{i}
  $$

  - 满足被忽略的约束条件 $\phi_{j} \geq 0$

#### 固定 $\phi, \ \sigma^{2}$ 对 $\mu$ 求导

- 忽略常数项，极大似然函数如下：

  $$
  L \left( \mu \right) = -\sum_{i = 1}^{m} \sum_{j = 1}^{k} w_{j}^{i} \cdot \frac{\left( x_{i} - \mu_{j} \right)^{2}}{2 \sigma_{j}^{2}} + C
  $$

- 对 $\mu_{j}$ 求导并令导数为 $0$：

  $$
  \mu_{j} = \frac{\sum_{i = 1}^{m} w_{j}^{i} \cdot x_{i}}{\sum_{i = 1}^{m} x_{i}}
  $$

#### 固定 $\phi, \ \mu$ 对 $t=\sigma^{2}$ 求导

- 忽略常数项，极大似然函数如下：

  $$
  L \left( t \right) = \sum_{i = 1}^{m} \sum_{j = 1}^{k} w_{j}^{i} \cdot \left( \log \frac{1}{\sqrt{2 \pi t_{j}}} - \frac{\left( x_{i} - \mu_{j}\right)^{2}}{2 t_{j}} \right) + C
  $$

- 对 $t_{j}$ 求导并令导数为 $0$：

  $$
  t_{j} = \frac{\sum_{i = 1}^{m} w_{j}^{i} \cdot \left( x_{i} - \mu_{j} \right)^{2}}{\sum_{i = 1}^{m} w_{j}^{i}}
  $$

- 将 $t=\sigma^{2}$ 代回上式：

  $$
  \sigma_{j}^{2} = \frac{\sum_{i = 1}^{m} w_{j}^{i} \cdot \left( x_{i} - \mu_{j} \right)^{2}}{\sum_{i = 1}^{m} w_{j}^{i}}
  $$

### 一般形式

#### $E$ 步

$$
w_{j}^{i} = \frac{\phi_{j} \cdot \sigma_{j}^{-1} \cdot \exp \left( \frac{\left( x_{i} - \mu_{j} \right)^{2}}{\sigma_{j}^{2}} \right)}{\sum_{l = 1}^{k} \phi_{l} \cdot \sigma_{l}^{-1} \exp \left( \frac{\left( x_{i} - \mu_{l} \right)^{2}}{\sigma_{l}^{2}} \right)}
$$

#### $M$ 步

$$
\phi_{j} = \frac{1}{m} \sum_{i = 1}^{m} w_{j}^{i}
$$

$$
\mu_{j} = \frac{\sum_{i = 1}^{m} w_{j}^{i} \cdot x_{i}}{\sum_{i = 1}^{m} x_{i}}
$$

$$
\sigma_{j}^{2} = \frac{\sum_{i = 1}^{m} w_{j}^{i} \cdot \left( x_{i} - \mu_{j} \right)^{2}}{\sum_{i = 1}^{m} w_{j}^{i}}
$$