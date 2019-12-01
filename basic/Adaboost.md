# $\mathrm{Adaboost}$

## 背景介绍

- 对于分类问题，训练弱分类器要比强分类器容易

- 从弱分类器出发，反复学习，得到一系列弱分类器，将这些弱分类器组合成一个强分类器

- $\mathrm{Adaboost}$ 提高错误分类样本的权重，降低正确分类样本的权重，权值的变动将使新的分类器更关注之前被错误分类的样本

- $\mathrm{Adaboost}$ 损失函数按指数速率下降

## 学习过程

- 假设样本集为 $\left\{ \left( x_{1}, \ y_{1} \right), \ \left( x_{2}, \ y_{2} \right), \ \cdots, \ \left( x_{M},\  y_{M} \right) \right\}$，样本标签 $y_{i} \in \left\{ -1, \ +1 \right\}$，样本的初始权重 $W_{1} = \left( \frac{1}{M}, \ \frac{1}{M}, \ \cdots, \ \frac{1}{M} \right)$

- 指示函数定义如下：

  $$
  I \left( c \right) = \left\{ \begin{matrix} 1, \quad \mathrm{if \ c \ is \ true} \\ 0, \quad \mathrm{if \ c \ is \ false} \end{matrix} \right.
  $$

- 进行 $T$ 次迭代，对于当前迭代次数 $t$：

  - 根据当前权重 $W_{t}$ 训练弱分类器 $h_{t} \left( x \right)$，并计算错误率 $e_{t}$：

    $$
    e_{t} = \sum_{i = 1}^{M} W_{t, \ i} \cdot I \ \left( h_{t} \left( x_{i} \right) \neq y_{i} \right)
    $$

  - 若 $e_{t} > 0.5$，结束迭代

  - 计算分类器 $h_{t} \left( x \right)$ 的权重 $\alpha_{t}$：

    $$
    \alpha_{t} = \frac{1}{2} \ \ln \left( \frac{1 - e_{t}}{e_{t}} \right)
    $$

  - 更新样本权重：

    $$
    W_{t + 1, \ i} = \frac{W_{t, \ i}}{Z_{t}} \cdot \mathrm{e}^{-\alpha_{t} \cdot y_{i} \cdot h_{t} \left( x_{i} \right)}
    $$

    - 其中 $Z_{t}$ 是归一化系数：

      $$
      Z_{t} = \sum_{i = 1}^{M} W_{t, \ i} \cdot \mathrm{e}^{-\alpha_{t} \cdot y_{i} \cdot h_{t} \left( x_{i} \right)}
      $$

- 得到最终的强分类器 $H \left( x \right)$：

  $$
  H \left( x \right) = \mathrm{sign} \left( \sum_{t = 1}^{T} \alpha_{t} \cdot h_{t} \left( x \right) \right)
  $$

## 数学推导

### 错误率 $e_{t}<0.5$

- 如果弱分类器的性能还不如随机猜测，那么学到的模型将毫无意义

### 归一化系数 $Z_{t}$

- 使 $W_{t}$ 成为一个概率分布

- 也是弱分类器 $h_{t} \left( x \right)$ 的损失函数

### 计算 $h_{t} \left( x \right)$ 权重 $\alpha_{t}$

#### 误差上限

- 当 $H \left( x_{i} \right) \neq y_{i}$ 时，$y_{i} H \left( x_{i} \right) = -1$，此时 $\mathrm{e}^{-y_{i} H \left( x_{i} \right)} > I \ \left( H \left( x_{i} \right) \neq y_{i} \right)$，误差上限如下：

  $$
  \frac{1}{M} \sum_{i = 1}^{M} I \ \left( H \left( x_{i} \right) \neq y_{i} \right) < \frac{1}{M} \sum_{i = 1}^{M} \mathrm{e}^{-y_{i} H \left( x_{i} \right)}
  $$

- 由每一轮样本权重更新公式可得：

  $$
  Z_{t} \cdot W_{t + 1, \ i} = W_{t, \ i} \cdot \mathrm{e}^{-\alpha_{t} \cdot y_{i} \cdot h_{t} \left( x_{i} \right)}
  $$

- 将上式代入误差上限可得：

  $$
  \begin{aligned}
  \frac{1}{M} \sum_{i = 1}^{M} \mathrm{e}^{-y_{i} H \left( x_{i} \right)} &= \sum_{i = 1}^{M} W_{1, \ i} \cdot \exp \left( \sum_{t = 1}^{T} -\alpha_{t} \cdot y_{i} \cdot h_{t} \left( x_{i} \right) \right) \newline
  &= \sum_{i = 1}^{M} W_{1, \ i} \cdot \prod_{t = 1}^{T} \exp \left( -\alpha_{t} \cdot y_{i} \cdot h_{t} \left( x_{i} \right) \right) \newline
  &= \sum_{i = 1}^{M} W_{1, \ i} \cdot \exp \left( -\alpha_{1} \cdot y_{i} \cdot h_{1} \left( x_{i} \right) \right) \cdot \prod_{t = 2}^{T} \exp \left( -\alpha_{t} \cdot y_{i} \cdot h_{t} \left( x_{i} \right) \right) \newline
  &= Z_{1} \cdot \sum_{i = 1}^{M} W_{2, \ i} \cdot \prod_{t = 2}^{T} \exp \left( -\alpha_{t} \cdot y_{i} \cdot h_{t} \left( x_{i} \right) \right) \newline
  &= \prod_{t = 1}^{T} Z_{t} \cdot \sum_{i = 1}^{M} W_{t + 1, \ i} \newline
  &= \prod_{t = 1}^{T} Z_{t} \newline
  \end{aligned}
  $$

- 因此，在每一轮训练弱分类器 $h_{t} \left( x \right)$ 时，应最小化归一化因子 $Z_{t}$

#### 系数 $\alpha_{t}$

- 对于每一个 $\alpha_{t}$，都要最小化损失函数 $Z_{t}$：

  $$
  \begin{aligned}
  Z_{t} &= \sum_{i = 1}^{M} W_{t, \ i} \cdot \mathrm{e}^{-\alpha_{t} \cdot y_{i} \cdot h_{t} \left( x_{i} \right)} \newline
  &= \sum_{y_{i} = h_{t} \left( x_{i} \right)} W_{t, \ i} \cdot \mathrm{e}^{-\alpha_{t}} + \sum_{y_{i} \neq h_{t} \left( x_{i} \right)} W_{t, \ i} \cdot \mathrm{e}^{\alpha_{t}} \newline
  &= \mathrm{e}^{-\alpha_{t}} \cdot \sum_{y_{i} = h_{t} \left( x_{i} \right)} W_{t, \ i} \cdot I \ \left( h_{t} \left( x_{i} \right) = y_{i} \right) + \mathrm{e}^{\alpha_{t}} \cdot \sum_{y_{i} \neq h_{t} \left( x_{i} \right)} W_{t, \ i} \cdot I \ \left( h_{t} \left( x_{i} \right) \neq y_{i} \right) \newline
  &= \mathrm{e}^{-\alpha_{t}} \cdot \left( 1 - e_{t} \right) + \mathrm{e}^{\alpha_{t}} \cdot e_{t} \newline
  \end{aligned}
  $$

- 计算 $\frac{\partial Z_{t}}{\partial \alpha_{t}}$ 并令偏导数为 $0$ 可得：

  $$
  \alpha_{t} = \frac{1}{2} \ \ln \left( \frac{1 - e_{t}}{e_{t}} \right)
  $$

#### 样本权重

- 将 $\alpha_{t}$ 代入 $Z_{t}$ 推导过程可得：

  $$
  Z_{t} = 2\sqrt{e_{t} \cdot \left( 1 - e_{t} \right)}
  $$

- 对于正确分类的样本：

  $$
  \frac{\mathrm{e}^{-\alpha_{t} \cdot y_{i} \cdot h_{t} \left( x_{i} \right)}}{Z_{t}} = \frac{\mathrm{e}^{-\alpha_{t}}}{Z_{t}} = \frac{1}{2 \cdot \left( 1 - e_{t} \right)} < 1
  $$

  - 在下一次迭代时，样本权重会减小

- 对于错误分类的样本：

  $$
  \frac{\mathrm{e}^{-\alpha_{t} \cdot y_{i} \cdot h_{t} \left( x_{i} \right)}}{Z_{t}} = \frac{\mathrm{e}^{\alpha_{t}}}{Z_{t}} = \frac{1}{2 \cdot e_{t}} > 1
  $$

  - 在下一次迭代时，样本权重会增大

#### 指数速率

- 令 $\gamma_{t} = \left( \frac{1}{2} - e_{t} \right) \in \left[ 0, \ 0.5 \right)$，代入 $Z_{t}$ 可得：

  $$
  Z_{t} = \sqrt{1 - 4\gamma_{t}^{2}}
  $$

- 由导数易知，$f \left( x \right) = \mathrm{e}^{-4 x^{2}} - \left( 1 - 4 x^{2} \right) \geq 0$ 在 $\left[ 0, \ 0.5 \right)$ 上恒成立，因此：

  $$
  \sqrt{1 - 4\gamma_{t}^{2}} \leq \mathrm{e}^{-2 \gamma_{t}^2}
  $$

- 将 $Z_{t}$ 与上述不等式代入误差上限可得：

  $$
  \prod_{t = 1}^{T} Z_{t} \leq \prod_{t = 1}^{T} \mathrm{e}^{-2 \gamma_{t}^2} \leq \exp \left( - \sum_{t = 1}^{T} 2 \gamma_{t}^2 \right)
  $$

- 因此 $\mathrm{Adaboost}$ 误差上限较小，同时还以指数速率下降