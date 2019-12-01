# $\mathrm{R-CNN}$

## 思想

- 将 $\mathrm{CNN}$ 引入目标检测领域

- 采用 $\mathrm{Selective \ Search}$ 减少搜索与计算

- 算法流程：

  - 选出一部分可能是目标物体的区域（$\mathrm{ROI}$）

  - 对 $\mathrm{CNN}$ 进行 $\mathrm{fine \ tuning}$ 并训练二分类 $\mathrm{SVM}$

  - 对 $\mathrm{ROI}$ 区域进行特征提取和分类

  - 得到各个目标物体的位置和类别

  - 对目标物体做 $\mathrm{Bounding \ Box}$ 回归，得到最终位置

- $\mathrm{ROI}$ 重叠导致 $\mathrm{CNN}$ 的重复计算，处理速度较慢：

  - $\mathrm{CPU}$：每张 $53$ 秒

  - $\mathrm{GPU}$：每张 $13$ 秒

## 候选区域选择

### 穷举式搜索

- 采用滑窗的方式，在整张图像上进行滑动，得到所有可能的区域

- 由于采用了穷举策略，时间复杂度较高

### $\mathrm{Selective \ Search}$

- 对图像进行分割，生成一些小的目标区域

- 计算这些区域与相邻区域的相似度，合并两个最相似的区域

- 重复上一步，直到剩下指定数量的区域（$2000$ 个）

## 特征提取与分类

- 对 $\mathrm{Selective \ Search}$ 的结果进行拉伸或填充，使其满足 $\mathrm{CNN}$ 的输入尺寸要求

- 对 $\mathrm{AlexNet}$、$\mathrm{VGG}$ 进行 $\mathrm{fine-tuning}$，修改最后的全连接层，对候选区域训练 $N + 1$ 类（前景、背景）分类器

- 用 $\mathrm{CNN}$ 提取候选区域的 $4096$ 维特征，通过二分类 $\mathrm{SVM}$，判断该特征是否属于某个类别

  - 在训练 $\mathrm{CNN}$ 时，为防止过拟合，进行了数据增强，对正样本定义比较宽松

  - 如果直接对提取的特征进行 $\mathrm{Softmax}$，分类效果不是很好

### $\mathrm{CNN \ fine-tuning}$

- 不进行 $\mathrm{fine-tuning}$ 时，使用 $\mathrm{fc6}$ 层特征效果较好

- 进行 $\mathrm{fine-tuning}$ 时，$\mathrm{fc6}$ 和 $\mathrm{fc7}$ 层特征的结果差距不大

- 相对于背景类，前景类样本数量较少；为平衡正负样本比例，需要放宽正样本的限制

  - 计算每个 $\mathrm{Region \ Proposal}$ 与 $\mathrm{Ground \ Truth}$ 的 $\mathrm{IoU}$

  - 将 $\mathrm{IoU} \geq 0.5$ 的 $\mathrm{Region \ Proposal}$ 视为正样本

  - 将 $\mathrm{IoU} < 0.5$ 的 $\mathrm{Region \ Proposal}$ 视为负样本

- 相对于背景框，前景目标较少；训练时每个 $\mathrm{batch}$ 内包含 $32$ 个正样本，$96$ 个负样本

### $\mathrm{SVM}$ 预训练

- 计算每个 $\mathrm{Region \ Proposal}$ 与 $\mathrm{Ground \ Truth}$ 的 $\mathrm{IoU}$

  - 只把 $\mathrm{Ground \ Truth}$ 视为正样本

  - 将 $\mathrm{IoU} < 0.3$ 的 $\mathrm{Region \ Proposal}$ 视为负样本（阈值通过实验确定）

  - 对于其他候选区域，不作为 $\mathrm{SVM}$ 训练的样本

## 非极大值抑制

- $\mathrm{Selective \ Search}$ 的结果可能互相重叠，需要从中选出得分最高的几个独立区域

- 因此对于每一类，按得分从高到低进行排序：

  - 只有当前区域与已选区域的重叠率不超过某个阈值，才保留该区域

## $\mathrm{Bounding \ Box}$ 回归

<center>
<img src="images/bounding_box.png"/>
</center>

### 目的

- 红色框 $P$ 是 $\mathrm{Region \ Proposal}$，绿色框 $G$ 是 $\mathrm{Bounding \ Box}$；由于偏差较大导致重叠率较小，相当于检测失败

- 可以对 $\mathrm{Region \ Proposal}$ 进行微调得到蓝色框 $\hat{G_{}}$，使其更接近 $\mathrm{Bounding \ Box}$，以提高检测精度

- 每个框都可以用 $\left( x, \ y, \ w, \ h \right)$ 表示，其中 $x, \ y$ 是中心坐标，$w, \ h$ 是宽和高

- $\mathrm{Bounding \ Box \ Regression}$ 的目标就是寻找一个映射 $f$：

  $$
  f \left( P_{x}, \ P_{y}, \ P_{w}, \ P_{h} \right) = \left( \hat{G_{x}}, \ \hat{G_{y}}, \ \hat{G_{w}}, \ \hat{G_{h}} \right)
  $$

  - 使下式成立：

    $$
    \left( \hat{G_{x}}, \ \hat{G_{y}}, \ \hat{G_{w}}, \ \hat{G_{h}} \right) \approx \left( G_{x}, \ G_{y}, \ G_{w}, \ G_{h} \right)
    $$

### 实现

- 为每一类训练一个 $\mathrm{Bounding \ Box}$ 回归器

- 实验结果表明，当 $\mathrm{Region \ Proposal}$ 与 $\mathrm{Ground \ Truth}$ 的 $\mathrm{IoU} \geq 0.6$ 时，作为 $\mathrm{Bounding \ Box}$ 的训练样本效果较好

- 需要学习的变换：$d_{x} \left( P \right), \ d_{y} \left( P \right), \ d_{w} \left( P \right), \ d_{h} \left( P \right)$：

  - $d_{*} \left( P \right)$ 框内物体特征 $\Phi \left( P \right)$ 的某个函数：

    $$
    d_{*} \left( P \right) = w_{*}^{T} \cdot \Phi \left( P \right)
    $$

#### 平移

$$
\hat{G_{x}} = P_{w} d_{x} \left( P \right) + P_{x}
$$

$$
\hat{G_{y}} = P_{h} d_{y} \left( P \right) + P_{y}
$$

#### 缩放

$$
\hat{G_{w}} = P_{w} \exp \left( d_{w} \left( P \right) \right)
$$

$$
\hat{G_{h}} = P_{h} \exp \left( d_{h} \left( P \right) \right)
$$

#### 回归

- 根据上述公式，$\mathrm{Region \ Proposal}$ 与 $\mathrm{Bounding \ Box}$ 平移量、缩放尺度计算如下：

  - 平移因子：

    $$
    t_{x} = \frac{G_{x} - P_{x}}{P_{w}}
    $$

    $$
    t_{y} = \frac{G_{y} - P_{y}}{P_{h}}
    $$

  - 缩放因子：

    $$
    t_{w} = \log \left( \frac{G_{w}}{P_{w}} \right)
    $$

    $$
    t_{h} = \log \left( \frac{G_{h}}{P_{h}} \right)
    $$

  - 由 $\hat{G_{}} \approx G$ 可得：

    $$
    t_{*} \approx d_{*} \left( P \right)
    $$

- 由于目标特征由同一个 $\mathrm{CNN}$ 提取，对 $t_{*}$ 的学习本质上是对 $w_{*}$ 的学习，正则化的损失函数如下：

  $$
  \mathrm{Loss} = \sum_{i = 1}^{N} \left( t_{*}^{\left( i \right)} - w_{*}^{T} \cdot \Phi \left( P^{i} \right) \right)^{2} + \lambda ||w_{*}||^{2}
  $$

  - 通过梯度下降或最小二乘即可求出 $w_{*}$ 的最优解