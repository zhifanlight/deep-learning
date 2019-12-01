# 广义线性模型

## 指数族分布

- 指数族分布具有以下形式：

  $$
  p \left( y; \ \eta \right) = b \left( y \right) \cdot \exp \left( \eta^{T}T \left( y \right) - a \left( \eta \right) \right)
  $$

  - $\eta$ 是自然参数

  - $T \left( y \right)$ 是 $y$ 的函数，通常 $T \left( y \right) = y$

  - $a \left( \eta \right)$ 是归一化因子，保证 $\sum p \left( y; \ \eta \right) = 1$

### 高斯分布

- 已知 $p \left( y; \ \phi \right) \sim N \left( \mu, \ \sigma^{2} \right)$，其中 $\phi = \left( \mu, \ \sigma^{2} \right)$

- 对于高斯分布，$b \left( y \right) = \frac{1}{\sqrt{2\pi} \sigma} \exp \left( -\frac{y^{2}}{2 \sigma^{2}} \right), \ \eta = \frac{\mu}{\sigma^{2}}, \ T \left( y \right) = y, \ a \left( \eta \right) = \frac{\eta^{2}\sigma^{2}}{2}$

  $$
  \begin{aligned}
  p \left( y; \ \phi \right) &= \frac{1}{\sqrt{2 \pi}\sigma} \exp \left( -\frac{\left( y - \mu \right)^2}{2 \sigma^{2}} \right) \newline
  &= \frac{1}{\sqrt{2 \pi}\sigma} \exp \left( \frac{2y \mu - y^{2} - \mu^{2}}{2 \sigma^{2}} \right) \newline
  &= \frac{1}{\sqrt{2 \pi}\sigma} \exp \left( -\frac{y^{2}}{2 \sigma^{2}} \right) \exp \left( \frac{\mu}{\sigma^{2}}y - \frac{\mu^{2}}{2\sigma^{2}}\right) \newline
  \end{aligned}
  $$

### 伯努利分布

- 已知 $p \left( y; \ \phi \right) \sim B \left( \phi \right)$，其中 $\phi$ 表示 $y = 1$ 的概率

- 对于伯努利分布，$b \left( y \right) = 1, \ \eta = \log \frac{\phi}{1 - \phi}, \ T \left( y \right) = y, \ a \left( \eta \right) = \log \left( 1 + \mathrm{e}^{\eta} \right)$

  $$
  \begin{aligned}
  p \left( y; \ \phi \right) &= \phi^{y} \cdot \left( 1 - \phi \right)^{1 - y} \newline
  &= \exp \left( \log \phi^{y} \cdot \left( 1 - \phi \right)^{1 - y} \right) \newline
  &= \exp \left( y \cdot \log \phi + \left( 1 - y \right) \cdot \log \left( 1 - \phi \right) \right) \newline
  &= \exp \left( y \cdot \log \frac{\phi}{1 - \phi} + \log \left( 1 - \phi \right) \right)
  \end{aligned}
  $$

  - 由 $\eta = \log \frac{\phi}{1 - \phi}$ 可导出 $\mathrm{sigmoid}$ 函数：

    $$
    \phi = \frac{1}{1 + \mathrm{e}^{-\eta}}
    $$

  - 将 $\phi$ 代入 $\log \left( 1 - \phi \right)$ 可得：

    $$
    p \left( y; \ \phi \right) = \exp \left( y \cdot \log \frac{\phi}{1 - \phi} - \log \left( 1 + \mathrm{e}^{\eta} \right) \right)
    $$

### 多项式分布

- 已知 $p \left( y; \ \phi \right) \sim M \left( \phi_{1}, \ \phi_{2}, \ \cdots, \ \phi_{K} \right)$，其中 $\phi_{i}$ 表示 $y = i$ 的概率

- 定义指示函数：

  $$
  I \left( c \right) = \left\{ \begin{matrix} 1, \quad \mathrm{if \ c \ is \ true} \\ 0, \quad \mathrm{if \ c \ is \ false} \end{matrix} \right.
  $$

- 对于多项式分布，$b \left( y \right) = 1, \ \eta = \left[ \begin{matrix} \log \frac{\phi_{1}}{\phi_{K}} \\ \log \frac{\phi_{2}}{\phi_{K}} \\ \vdots \\ \log \frac{\phi_{K - 1}}{\phi_{K}} \end{matrix} \right], \ T \left( y \right) = \left[ \begin{matrix} I \left( y = 1 \right) \\ I \left( y = 2 \right) \\ \vdots \\ I \left( y = K - 1 \right) \end{matrix} \right], \ a \left( \eta \right) = \log \left( \sum_{i = 1}^{K} \mathrm{e}^{\eta_{i}} \right)$

  $$
  \begin{aligned}
  p \left( y; \ \phi \right) &= \prod_{i = 1}^{K} \phi_{i}^{I \left( y = i \right)} \newline
  &= \exp \left( \log \prod_{i = 1}^{K} \phi_{i}^{I \left( y = i \right)} \right) \newline
  &= \exp \left( \sum_{i = 1}^{K} I \left( y = i \right) \cdot \log \ \phi_{i} \right) \newline
  &= \exp \left( \sum_{i = 1}^{K - 1} \left( {I \left( y = i \right)} \cdot \log \phi_{i} \right) + \left( 1 - \sum_{i = 1}^{K - 1} I \left( y = i \right) \right) \cdot \log \phi_{K} \right) \newline
  &= \exp \left( \sum_{i = 1}^{K - 1} \left( {I \left( y = i \right)} \cdot \log \frac{\phi_{i}}{\phi_{K}} \right) + \log \phi_{K} \right) \newline
  \end{aligned}
  $$

  - 令 $\eta_{K} = 0$，由 $\eta = \left[ \begin{matrix} \log \frac{\phi_{1}}{\phi_{K}} \\ \log \frac{\phi_{2}}{\phi_{K}} \\ \vdots \\ \log \frac{\phi_{K - 1}}{\phi_{K}} \end{matrix} \right]$ 可推导出：

    $$
  \sum_{i = 1}^{K - 1} \mathrm{e}^{\eta_{i}} = \frac{1 - \phi_{K}}{\phi_{k}} \quad \Rightarrow \quad \phi_{K} = \frac{1}{1 + \sum_{i = 1}^{K - 1} \mathrm{e}^{\eta_{i}}} = \frac{1}{\sum_{i = 1}^{K} \mathrm{e}^{\eta_{i}}}
    $$

  - 将 $\phi_{K}$ 代入 $\log \phi_{K}$ 可得：

    $$
    p \left( y; \ \phi \right) = \exp \left( \sum_{i = 1}^{K - 1} \left( {I \left( y = i \right)} \cdot \log \frac{\phi_{i}}{\phi_{K}} \right) - \log \left( \sum_{i = 1}^{K} \mathrm{e}^{\eta_{i}} \right) \right)
    $$

## 广义线性模型

### 模型假设

- $y|x; \ \theta \sim \exp \left( \eta \right)$，即对于给定的 $x$ 和 $\theta$，$y$ 服从以 $\eta$ 为参数的指数族分布

- 给定 $x$，广义线性模型的求解目标是 $h_{\theta} \left( x \right) = \mathbb{E} \left[ T \left( y \right)|x \right]$

- 自然参数 $\eta$ 与 $x$ 是线性关系：$\eta = \theta^{T} x$

### 线性回归

- 假设 $y|x; \ \theta \sim N \left( \mu, \ \sigma^{2} \right)$，由高斯分布对应的广义线性模型：

  $$
  h_{\theta} \left( x \right) = \mathbb{E} \left[ T \left( y \right)|x \right] = \mathbb{E} \left[ y|x \right] = \mu = \eta \cdot \sigma^{2}
  $$

- 由 $\eta$ 与 $x$ 的线性关系可得：

  $$
  h_{\theta} \left( x \right) = \theta^{T} x \cdot \sigma^{2}
  $$

- 对于给定高斯分布，$\sigma^{2}$ 固定但未知，取 $\sigma^{2} = 1$：

  $$
  h_{\theta} \left( x \right) = \theta^{T} x
  $$

### 逻辑回归

- 假设 $y|x; \ \theta \sim B \left( \phi \right)$，由伯努利分布对应的广义线性模型：

  $$
  h_{\theta} \left( x \right) = \mathbb{E} \left[ T \left( y \right)|x \right] = \mathbb{E} \left[ y|x \right] = \phi = \frac{1}{1 + \mathrm{e}^{-\eta}}
  $$

- 由 $\eta$ 与 $x$ 的线性关系可得：

  $$
  h_{\theta} \left( x \right) = \frac{1}{1 + \mathrm{e}^{-\theta^{T} x}}
  $$

### $\mathrm{softmax}$ 回归

- 假设 $y|x; \ \theta \sim M \left( \phi_{1}, \ \phi_{2}, \ \cdots, \ \phi_{K} \right)$，由多项式分布对应的广义线性模型：

  $$
  h_{\theta} \left( x \right) = \mathbb{E} \left[ T \left( y \right)|x \right] = \left[ \begin{matrix} \phi_{1} \\ \phi_{2} \\ \vdots \\ \phi_{K - 1} \end{matrix} \right] = \left[ \begin{matrix} \mathrm{e}^{\eta_{1}} \cdot \phi_{K} \\ \mathrm{e}^{\eta_{2}} \cdot \phi_{K} \\ \vdots \\ \mathrm{e}^{\eta_{K - 1}} \cdot \phi_{K} \end{matrix} \right] = \left[ \begin{matrix} \frac{\exp \left( \eta_{1} \right)}{\sum_{j = 1}^{K} \exp \left( \eta_{j} \right)} \\ \frac{\exp \left( \eta_{2} \right)}{\sum_{j = 1}^{K} \exp \left( \eta_{j} \right)} \\ \vdots \\ \frac{\exp \left( \eta_{K - 1} \right)}{\sum_{j = 1}^{K} \exp \left( \eta_{j} \right)} \end{matrix} \right]
  $$

- 由 $\eta$ 与 $x$ 的线性关系可得：

  $$
  h_{\theta} \left( x \right) = \left[ \begin{matrix} \frac{\exp \left( \theta_{1}^{T} x \right)}{\sum_{j = 1}^{K} \exp \left( \theta_{j}^{T} x \right)} \\ \frac{\exp \left( \theta_{2}^{T} x \right)}{\sum_{j = 1}^{K} \exp \left( \theta_{j}^{T} x \right)} \\ \vdots \\ \frac{\exp \left( \theta_{K - 1}^{T} x \right)}{\sum_{j = 1}^{K} \exp \left( \theta_{j}^{T} x \right)} \end{matrix} \right]
  $$

- 当 $K = 2$ 时，$\mathrm{softmax}$ 回归退化为逻辑回归：

  $$
  h_{\theta} \left( x \right) = \left[ \frac{\mathrm{e}^{\theta_{1}^{T} x}}{\mathrm{e}^{\theta_{1}^{T} x} + \mathrm{e}^{\theta_{2}^{T} x}} \right] = \frac{\mathrm{e}^{\theta_{1}^{T} x}}{\mathrm{e}^{\theta_{1}^{T}x} + 1} = \frac{1}{1 + \mathrm{e}^{-\theta_{1}^{T} x}}
  $$