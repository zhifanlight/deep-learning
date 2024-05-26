# $\mathrm{DDIM}$

## 基本思想

- $\mathrm{DDIM}$ 是 $\mathrm{DDPM}$ 的扩展，$\mathrm{DDPM}$ 是 $\mathrm{DDIM}$ 的一个特例

- 前向扩散过程和反向去噪过程，不再依赖马尔可夫假设

- 与 $\mathrm{DDPM}$ 具有相同的训练目标，因此可以直接利用训练好的 $\mathrm{DDPM}$ 模型，不用重新训练

- 在 $\mathrm{DDIM}$ 的去噪过程中，每一步采样如下（证明略）：

  $$
  x_{t - 1} = \sqrt{\bar{\alpha}_{t - 1}} \left( \frac{x_{t} - \sqrt{1 - \bar{\alpha}}_{t} \ \hat{\epsilon}_{t} \left( x_{t}, \ t \right)}{\sqrt{\bar{\alpha}_{t}}} \right) + \sqrt{1 - \bar{\alpha}_{t - 1} - \sigma_{t}^{2}} \ \ \hat{\epsilon_{t}} \left( x_{t}, \ t \right) + \sigma_{t} \ \epsilon_{t}^{*}
  $$

  - 其中 $\epsilon_{t}^{*} \sim N \left( 0, \ I \right) $ 是标准高斯噪声

- 相比 $\mathrm{DDPM}$，$\mathrm{DDIM}$ 去噪过程多了一个方差参数 $\sigma^{2}$，实际上是 $q_{\sigma} \left( x_{t - 1} \mid x_{t}, \ x_{0} \right)$ 的方差

- 当 $\sigma^{2} = \frac{\left( 1 - \alpha_{t} \right) \left( 1 - \bar{\alpha}_{t - 1}\right)}{1 - \bar{\alpha}_{t}}$ 时，$\mathrm{DDIM}$ 退化成 $\mathrm{DDPM}$

- 当 $\sigma^{2} = 0$ 时，去噪采样过程变成一个确定的过程，不再具有随机性（方差为 $0$）

## 加速采样

- 当 $\sigma^{2} = 0$ 时，相当于每次都在 $q_{\sigma} \left( x_{t - 1} \mid x_{t}, \ x_{0} \right)$ 的概率密度最大点处进行采样，可以加速采样

  - 相比 $\mathrm{DDPM}$ 的 $1000$ 步采样，$\mathrm{DDIM}$ 可以提速 $10-100$ 倍，采样 $20-50$ 步即可完成去噪过程

- 更进一步，可以将 $\sigma^{2}$ 作为可调节的参数

  - 比如在训练开始时刻令 $\sigma^{2}=0$ 加速收敛，在最后时刻令 $\sigma^{2} \neq 0$ 以增加多样性

## 隐变量

- 在推导过程中，与 $\mathrm{DDPM}$ 对隐变量的处理方式不同，解除了隐变量的马尔可夫依赖