# $\mathrm{EfficientNet}$

## 背景

- 对于同系列网络，常用的精度提升方法是加深、加宽、提高分辨率

  - 但是，单独提升上述某一项，网络性能很容易陷入饱和

  - 之前的工作中，上述三项很少同时使用，而且缺乏理论指导

- 直觉上，如果提高分辨率，相应的网络应更深、更宽

  - 更深的网络具有更大的感受野，以匹配更大的分辨率

  - 更宽的网络具有更强的特征提取能力，以匹配更多的像素点

- 因此，最好是同时加深、加宽、提高分辨率

## 思想

- 在计算量翻倍的情况下，深度 $d$、宽度 $w$、分辨率 $r$ 分别按如下规则进行调整：

  $$
  d = \alpha ^ {\phi} \qquad w = \beta ^ {\phi} \qquad r = \gamma ^ {\phi}
  $$

  - 其中，$\alpha, \ \beta, \ \gamma$ 满足以下约束：

    $$
    \alpha \cdot \beta ^ {2} \cdot \gamma ^ {2} \approx 2
    $$

    - 上述约束基于：深度与计算量是线性关系，而宽度、分辨率与计算量是二次关系

## 网络

- 通过网络结构方法得到基础的 $\mathrm{EfficientNet-B0}$

  - 不针对任何特定平台，网络搜索的优化目标是计算量本身

  - 基本单元类似 $\mathrm{MobileNet-v2}$，使用 $\mathrm{SE}$ 模块和 $\mathrm{Swish}$ 激活函数

- 通过网格搜索确定 $\alpha, \ \beta, \ \gamma$ 的最优值如下：

  $$
  \alpha = 1.2 \qquad \beta = 1.1 \qquad \gamma=1.15
  $$

- 按照上述规则，不断扩展网络的计算量，得到 $\mathrm{EfficientNet-B1}$ 至 $\mathrm{EfficientNet-B7}$ 的系列网络