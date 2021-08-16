# $\mathrm{EfficientNet}$

## $\mathrm{EfficientNet \ v1}$

### 背景

- 对于同系列网络，常用的精度提升方法是加深、加宽、提高分辨率

  - 但是，单独提升上述某一项，网络性能很容易陷入饱和

  - 之前的工作中，上述三项很少同时使用，而且缺乏理论指导

- 直觉上，如果提高分辨率，相应的网络应更深、更宽

  - 更深的网络具有更大的感受野，以匹配更大的分辨率

  - 更宽的网络具有更强的特征提取能力，以匹配更多的像素点

- 因此，最好是同时加深、加宽、提高分辨率

### 思想

- 在计算量翻倍的情况下，深度 $d$、宽度 $w$、分辨率 $r$ 分别按如下规则进行调整：

  $$
  d = \alpha ^ {\phi} \qquad w = \beta ^ {\phi} \qquad r = \gamma ^ {\phi}
  $$

  - 其中，$\alpha, \ \beta, \ \gamma$ 满足以下约束：

    $$
    \alpha \cdot \beta ^ {2} \cdot \gamma ^ {2} \approx 2
    $$

    - 上述约束基于：深度与计算量是线性关系，而宽度、分辨率与计算量是二次关系

### 网络

- 通过网络结构方法得到基础的 $\mathrm{EfficientNet \ v1-B0}$

  - 不针对任何特定平台，网络搜索的优化目标是计算量本身

  - 基本单元类似 $\mathrm{MobileNet-v2}$，使用 $\mathrm{SE}$ 模块和 $\mathrm{Swish}$ 激活函数

- 通过网格搜索确定 $\alpha, \ \beta, \ \gamma$ 的最优值如下：

  $$
  \alpha = 1.2 \qquad \beta = 1.1 \qquad \gamma=1.15
  $$

- 按照上述规则，不断扩展网络的计算量，得到 $\mathrm{EfficientNet \ v1-B1}$ 至 $\mathrm{EfficientNet \ v1-B7}$ 的系列网络

### $\mathrm{EfficientNet \ v1-Lite}$

- $\mathrm{EfficientNet \ v1-Lite}$ 是针对边缘设备优化的版本，主要包括以下改进：

  - 移除 $\mathrm{SE}$ 模块，原因是边缘设备对 $\mathrm{SE}$ 模块的支持欠佳

  - 使用 $\mathrm{ReLU6}$ 代替 $\mathrm{Swish}$ 函数，提升训练后的量化质量

  - 模型缩放时，固定 $\mathrm{stem}$ （网络起始层）和 $\mathrm{head}$ （分类层及前一层），以减少模型体积及计算量

## $\mathrm{EfficientNet \ v2}$

- 分析并改进 $\mathrm{EfficientNet \ v1}$ 中存在的问题

- 相比 $\mathrm{EfficientNet \ v1}$ 及其他 $\mathrm{SOTA}$ 模型，具有更快的训练速度，更高效的参数利用率

### 训练图片尺寸、训练速度、内存/显存占用

- 训练图片太大时，训练速度很慢，内存/显存占用很大

  - 使用小 $\mathrm{batch}$ 虽然可以解决内存/显存占用问题，但会降低训练速度

  - 而使用小图片，既可以解决内存/显存占用问题，又可以加速训练，还可以轻微提升精度（实验结果）

- 通过由小到大，动态调整训练图片尺寸，可以进一步提升训练速度

  - 为解决之前工作中，动态调节图片尺寸导致的精度下降问题，提出动态数据增强、正则化约束

    - 对于小图片，使用较弱的数据增强和正则化约束

    - 对于大图片，使用较强的数据增强和正则化约束

    - 整个训练过程中，由小到大动态调整图片尺寸，动态加强数据增强、正则化约束

### $\mathrm{Depthwise}$ 卷积在浅层网络中的速度问题

- $\mathrm{Depthwise}$ 虽然理论计算量更低，但受限于软硬件，实际运行速度可能会更慢；而 $\mathrm{Fused-MBConv}$ 可以更充分地利用手机等硬件的加速器

- $\mathrm{Fused-MBConv}$ 是指将 $\mathrm{MobileNet \ v2 \ Block}$ 的第一个 $\mathrm{1 \times 1 \ Pointwise}$ 卷积、$\mathrm{3 \times 3 \ Depthwise}$ 卷积替换为普通的 $\mathrm{3 \times 3}$ 卷积

  - 只替换 $\mathrm{EfficientNet \ v1}$ 的前 $3$ 个 $\mathrm{stage}$，可以提升训练速度，参数量和计算量几乎不变

  - 替换 $\mathrm{EfficientNet \ v1}$ 的所有 $\mathrm{stage}$，会导致参数量、计算量剧增，同时影响训练速度和精度

  - 具体要替换前几层，由 $\mathrm{NAS}$ 结果决定

### 合理设置每个 $\mathrm{stage}$ 的放大比例

- 每个 $\mathrm{stage}$ 对训练速度、参数量的影响不同，等比例放大每个 $\mathrm{stage}$ 不是最优解

- 在网络深处（$\mathrm{stage5, \ stage6}$）添加更多的层，可以提高模型容量，但不会明显影响运行速度

### 网络结构变更（$\mathrm{NAS}$ 结果）

- 浅层使用 $\mathrm{Fused-MBConv}$，深层使用 $\mathrm{MobileNet \ v2 \ Block}$

- 在 $\mathrm{MobileNet \ v2 \ Block}$ 中倾向于使用更小的扩张系数，内存/显存访存更高效

- 倾向于使用 $\mathrm{3 \times 3}$ 小卷积核，但为了维持感受野，可能会堆叠更多的卷积层

- 移除 $\mathrm{EfficientNet \ v1}$ 中最后一个 $\mathrm{stride=1}$ 的 $\mathrm{stage}$，可能是为了降低参数量和速度损耗

### 其它贡献

- 对于在 $\mathrm{ImageNet-1k}$ 上 $\mathrm{Top-1}$ 超过 $\mathrm{85\%}$ 的模型，扩大数据集比扩大模型尺寸，更容易提升精度