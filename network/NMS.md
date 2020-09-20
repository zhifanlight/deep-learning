# $\mathrm{NMS}$

## $\mathrm{NMS}$

### 背景介绍

- 在检测任务中，得到的检测框大多相互重叠，使用非极大值抑制（$\mathrm{Non \ Maximum \ Suppression}$，即 $\mathrm{NMS}$）可以去除大多数重叠框，只保留置信度最高的候选框

### 计算过程

#### 排序

- 首先，对所有 $\mathrm{BBox}$ 集合 $B$ 按置信度降序排序

- 依次将置信度最高的 $\mathrm{BBox}$ 加入集合 $P$ 并从 $B$ 中删除，按以下规则处理依次处理 $B$ 中其余 $\mathrm{BBox}$：

  - 如果当前 $\mathrm{BBox}$ 与 $P$ 中所有 $\mathrm{BBox}$ 的 $\mathrm{IoU}$ 都不超过给定阈值，将该 $\mathrm{BBox}$ 加入集合 $P$，并从 $B$ 中删除

  - 否则，直接将其从 $B$ 中删除

#### 不排序

- 与排序方法的过程基本相同

- 具体实现时，可以不借助额外的集合 $P$，直接将集合 $B$ 分为前后两部分：已选 $\mathrm{BBox}$ 及未选 $\mathrm{BBox}$，并记录已选个数 $N$；最终返回集合的前 $N$ 个元素。集合初始容量为 $\mathrm{BBox}$ 个数 $M$

  - 每次选取置信度最高的 $\mathrm{BBox}$ 时，需要遍历所有未选元素，并与当前位置交换

  - 遍历剩余 $\mathrm{BBox}$ 时，如果某个 $\mathrm{BBox}$ 与已选的某个 $\mathrm{BBox}$ 的 $\mathrm{IoU}$ 超过给定阈值，直接删除

    - 实现方式：将其与当前集合末尾元素交换，并将集合容量 $M$ 减 $1$

### 性能分析

- 可以去除大部分的重叠框，检测结果更干净

- 但如果在某个区域确实存在两个重合度较高的目标（遮挡情况），由于 $\mathrm{NMS}$ 直接删除置信度较低的目标，会造成检测器性能下降

## $\mathrm{Soft-NMS}$

### 提出背景

- 针对遮挡问题对 $\mathrm{NMS}$ 进行改进：对于重合度较高的两个 $\mathrm{BBox}$，不是直接删除低置信度的 $\mathrm{BBox}$，而是根据二者的 $\mathrm{IoU}$ 动态降低该 $\mathrm{BBox}$ 的置信度

### 计算过程

- 为了统一 $\mathrm{NMS}$ 和 $\mathrm{Soft-NMS}$ 的计算过程，可以将 $\mathrm{NMS}$ 的删除操作视为将置信度置 $0$

  - 与 $\mathrm{NMS}$（不排序方式） 基本相同

  - 不同点：对重叠的 $\mathrm{BBox}$，不直接将置信度置 $0$，而是根据 $\mathrm{IoU}$ 降低

  - 如果 $\mathrm{BBox}$ 的置信度依旧低于阈值，将其从 $\mathrm{BBox}$ 集合中删除

- 通常使用高斯加权的方式降低置信度：

  $$
  s_{i} = s_{i} \exp \left( - \frac{\mathrm{iou} \left( M, b_{i} \right)^{2}}{\sigma} \right)
  $$

  - 其中，$\sigma$ 是超参数，通常取 $0.5$

- 也可以通过线性加权的方式降低置信度，但效果略差：

  $$
  s_{i} = \left\{ \begin{matrix} s_{i} \\ s_{i} \left( 1 - \mathrm{iou} \left( M, b_{i} \right) \right) \end{matrix} \right. \qquad \begin{matrix} \mathrm{iou} \left( M, b_{i} \right) \lt T \\ \mathrm{iou} \left( M, b_{i} \right) \ge T \end{matrix}
  $$

### 性能分析

- 作为 $\mathrm{NMS}$ 的替代品，精度提高 $1-2$ 个点，速度不变

- 无需重新训练，即插即用

- 同样基于贪心策略，无法进行全局最优的置信度重置