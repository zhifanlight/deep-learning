# 模型评价

## 评估方法

### 留出法

- 直接将数据集划分为两个互斥的训练集、验证集

  - 训练集、验证集的划分要尽可能保持数据一致性

  - 训练集、验证集比例介于 $2:1 - 4:1$

- 在训练集上训练出模型后，计算验证集上的泛化误差

  - 一般要进行多次随机划分，将平均值作为最终的泛化误差

### 交叉验证

- 将数据集分为 $k$ 个大小相似的互斥子集

  - 每个子集尽可能保持分布的一致性

- 每次选择 $1$ 个子集作为验证集，其他 $k - 1$ 个子集作为训练集，计算泛化误差

  - 计算 $k$ 个验证集上误差的平均值，作为最终的泛化误差

- 当 $k$ 等于样本数时，称为“留一法”，训练时计算开销较大

- $k$ 一般取 $10$

### $\mathrm{BootStrap}$

- 留出法和交叉验证在训练时，只使用了数据集的一个子集

- 对于有 $m$ 个样本的数据集，随机有放回采样 $m$ 次，得到训练集

- 某个样本在 $m$ 次采样中均未被选中的概率为

  $$
  \lim_{m \rightarrow \infty} \left( 1 - \frac{1}{m} \right) ^{m} = \frac{1}{\mathrm{e}} = 0.368
  $$

  - 约有 $1 \ / \ 3$ 的数据未被选中，将这些数据作为验证集

- $\mathrm{BootStrap}$ 能从原始数据集产生多个不同的训练集

## 正确率

- $\mathrm{Accuracy}$

- 正确分类的样本数占样本总数的比例

## 查准率

- $\mathrm{Precision}$，又称精确率

- 预测出的正样本中，实际正样本所占比例

## 查全率

- $\mathrm{Recall}$，又称召回率

- 预测出的正样本中，实际的正样本在所有正样本中的比例

## $F$ 值

- $\frac{1}{F} = \alpha \frac{1}{P} + \left( 1 - \alpha \right) \frac{1}{R}$，即 $F = \frac{\left( \beta^{2} + 1 \right) PR}{\beta^{2} P + R}$，而 $\beta^{2} = \frac{1 - \alpha}{\alpha}$

- 其中 $\alpha$ 或 $\beta$ 决定查准率 $P$ 和查全率 $R$ 的权重：$\alpha > 0.5$ 或 $\beta < 1$ 时强调查准率 $P$；$\alpha < 0.5$ 或 $\beta > 1$ 时强调查全率 $R$

- $\beta = 1$ 或 $\alpha = 0.5$ 时，查准率 $P$ 和查全率 $R$ 的权重相同，此时的的 $F$ 值称为 $F_{1}$ 值

- $F_{1}$ 值综合了查准率 $P$ 和查全率 $R$，当 $F_{1}$ 值较高时，说明结果比较理想

## 真正例率

- $\mathrm{TPR}$，即 $\mathrm{True \ Positive \ Rate}$

- 预测出的正样本中，实际的正样本在所有正样本中的比例；也就是查全率

- 表示实际正样本中被正确预测的概率

## 假正例率

- $\mathrm{FPR}$，即 $\mathrm{False \ Positive \ Rate}$

- 预测出的正样本中，实际的负样本在所有负样本中的比例

- 表示实际负样本中被错误预测为正样本的概率

## $P-R$ 曲线

- $\mathrm{Precision-Recall}$，即 查准率-查全率 曲线

- 通过调整不同的阈值区分正负样本，得到一组 $\left( R, \ P \right)$ 值，将这些 $\left( R, \ P \right)$ 坐标绘制成 $P-R$ 曲线

- 曲线越靠近右上角，整体效果越好

- $P-R$ 曲线不单调，因此可以用来参数调优

- 正负样本分布变化时，$P-R$ 曲线会发生较大变化

- 如果分类器 $A$ 的 $P-R$ 曲线包含分类器 $B$ 的 $P-R$ 曲线，那么分类器 $A$ 的性能更好

<center>
<img src="images/pr.png"/>
</center>

## $\mathrm{AP \ \& \ mAP}$

- $P-R$ 曲线下的面积称为 $\mathrm{AP}$，即 $\mathrm{Average \ Precision}$。$\mathrm{AP}$ 越大，检索性能越好

- 对多个类别的 $\mathrm{AP}$ 值进行平均得到 $\mathrm{mAP}$，即 $\mathrm{mean \ Average \ Precision}$

### 计算方式

- 设目标检索结果是 $\{D_{1}, \ D_{2}, \ \cdots, \ D_{n} \}$，$R_{i}$ 是检索到 $D_{i}$ 时的排名，则：

  $$
  \mathrm{AP} = \frac{1}{n} \sum_{i = 1}^{n} \frac{i}{R_{i}}
  $$

- 对于以下检索结果，$\mathrm{AP}$ 值为：

  $$
  \mathrm{AP} = \frac{1}{3} \cdot \left( \frac{1}{1} + \frac{2}{3} + \frac{3}{6} \right) = 0.722
  $$

  <center>
  <img src="images/ap.png"/>
  </center>

## $\mathrm{ROC}$ 曲线

- 通过调整不同的阈值区分正负样本，得到一组 $\left( \mathrm{FPR}, \ \mathrm{TPR} \right)$ 值，将这些 $\left( \mathrm{FPR}, \ \mathrm{TPR} \right)$ 坐标绘制成 $\mathrm{ROC}$ 曲线

- $\mathrm{ROC}$ 曲线的纵坐标为 $P-R$ 曲线的横坐标

- 曲线越靠近左上角，整体效果越好

- $\mathrm{ROC}$ 曲线单调递增，衡量的是分类器的相对性能

- 单调性不完全证明：

  - $\mathrm{TPR} = \frac{\mathrm{TP}}{\mathrm{TP} + \mathrm{FN}} = \frac{1}{1 + \frac{\mathrm{FN}}{\mathrm{TP}}}$，$\mathrm{FPR} = \frac{\mathrm{FP}}{\mathrm{FP} + \mathrm{TN}} = \frac{1}{1 + \frac{\mathrm{TN}}{\mathrm{FP}}}$

  - 当样本集固定时，$\mathrm{TP + FN}$、$\mathrm{FP + TN}$ 均为定值

  - 当阈值减小时，$\mathrm{TP}$、$\mathrm{FP}$ 都增大，导致 $\mathrm{FN}$、$\mathrm{TN}$ 都减小，$\mathrm{TPR}$、$\mathrm{FPR}$ 都增大

  - 当阈值为 $1$ 时，所有样本都被预测为负样本，此时 $\mathrm{TPR} = \mathrm{FPR} = 0$，即 $\left( 0, \ 0 \right)$ 点

  - 当阈值为 $0$ 时，所有样本都被预测为正样本，此时 $\mathrm{TPR} = \mathrm{FPR} = 1$，即 $\left( 1, \ 1 \right)$点

  - 直观上，从左下角过渡到右上角，阈值越来越低；因此，$\mathrm{ROC}$ 曲线单调递增

- 正负样本分布变化时，$\mathrm{ROC}$ 曲线保持不变。证明：

  - 由 $\mathrm{TPR}$、$\mathrm{FPR}$ 意义可知：无论正负样本如何分布，相同阈值下的 $\mathrm{TPR}$、$\mathrm{FPR}$ 不变

- 如果分类器 $A$ 的 $\mathrm{ROC}$ 曲线包含分类器 $B$ 的 $\mathrm{ROC}$ 曲线，那么分类器 $A$ 的性能更好

<center>
<img src="images/roc.png"/>
</center>

## $\mathrm{AUC}$

- $\mathrm{ROC}$ 曲线下的面积称为 $\mathrm{AUC}$，即 $\mathrm{Area \ Under \ Curve}$。$\mathrm{AUC}$ 越大，分类器性能越好

  - $\mathrm{AUC} = 1$，完美分类器

    - 所有正样本都被被预测为正样本，没有负样本被预测为正样本，此时 $\mathrm{AUC} = 1.0$，即 $\mathrm{TPR} = 1.0$，$\mathrm{FPR} = 0.0$ 恒成立

  - $0.5 < \mathrm{AUC} < 1$，强于随机猜测

  - $\mathrm{AUC} = 0.5$，与随机猜测效果一样

    - 从所有样本中各选择一半正负样本，将其预测为正样本，此时 $\mathrm{AUC} = 0.5$

- 假设分类器输出的是对正类的置信度 $\mathrm{score}$，$\mathrm{AUC}$ 的物理意义是：

  - 任取一对正、负样本，正样本 $\mathrm{score}$ 大于负样本的概率

### 计算方式

- 假设有 $M$ 个正样本，$N$ 个负样本，共有 $T = M + N$ 个样本

- 按得分对样本进行排序，对于所有 $M \cdot N$ 个正、负样本对，统计正样本得分高于负样本的样本对数量 $S$，则：

  $$
  \mathrm{AUC} = \frac{S}{M \cdot N}
  $$

- 为加速计算，可令得分最高的样本 $\mathrm{rank}$ 为 $T$，第二高的样本为 $T - 1$，以此类推：

  - 假设得分最高的正样本 $\mathrm{rank}$ 为 $R_{1}$，那么共有 $\left( R_{1} - 1 \right) - \left( M - 1 \right) = R_{1} - M$ 个负样本的得分比该正样本低

  - 假设得分第二高的正样本 $\mathrm{rank}$ 为 $R_{2}$，那么共有 $\left( R_{2} - 1 \right) - \left( M - 2 \right) = R_{2} - \left( M - 1 \right)$ 个负样本的得分比该正样本低

  - 假设得分最低的正样本 $\mathrm{rank}$ 为 $R_{M}$，那么共有 $\left( R_{M} - 1 \right) - \left( M - M \right) = R_{M} - 1$ 个负样本的得分比该正样本低

  - 因此：

    $$
    S = \sum_{i = 1}^{M} R_{i} - \sum_{j = 1}^{M} j = \sum_{i = 1}^{M} R_{i} - \frac{M \left( M + 1 \right)}{2}
    $$

## $\mathrm{IoU}$

- $\mathrm{Intersection \ over \ Union}$

- 目标检测中常用的衡量指标，计算 $\mathrm{Region \ Proposal}$ 与 $\mathrm{Bounding \ Box}$ 的重叠率

- 假设 $\mathrm{Region \ Proposal}$ 区域为 $P$，$\mathrm{Bounding \ Box}$ 区域为 $B$，计算如下：

  $$
  \mathrm{IoU} = \frac{P \cap B}{P \cup B}
  $$

- $\mathrm{IoU}$ 越大，检测结果越准确；当 $\mathrm{IoU} = 1$ 时，$\mathrm{Region \ Proposal}$ 和 $\mathrm{Bounding \  Box}$ 完全重合