# 基本理论

## 统计学习三要素

### 模型

- 模型所要学习的条件概率分布、决策函数

- 模型的输入空间、输出空间

### 策略

- 经验风险最小化

  $$
  \min_{f \in F} \frac{1}{N} \sum_{i = 1}^{N} L \left( y_{i}, \ f \left( x_{i} \right) \right)
  $$

- 结构风险最小化

  $$
  \min_{f \in F} \frac{1}{N} \sum_{i = 1}^{N} L \left( y_{i}, \ f \left( x_{i} \right) \right) + \lambda \cdot J \left( f \right)
  $$

### 算法

- 学习模型的具体计算方法

## 偏差—方差分解

- 以回归任务为例，假设样本为 $x$，真实标记为 $y$，在训练集中的标记为 $y_{D}$，训练集 $D$ 上学得模型的预测输出为 $f \left( x; \ D \right)$

- 学习算法的期望输出：

  $$
  \bar{f} \left( x \right) = \mathbb{E}_{D} \left[ f \left( x; \ D \right) \right]
  $$

- 使用不同训练集产生的方差：

  $$
  \mathrm{var} \left( x \right) = \mathbb{E}_{D} \left[ \left( f \left( x; \ D \right) - \bar{f} \left( x \right) \right)^{2} \right]
  $$

  - 方差衡量了数据集变动导致的学习性能的变化

- 由于标记问题产生的噪声：

  $$
  \epsilon^{2} = \mathbb{E}_{D} \left[ \left( y_{D} - y \right)^{2} \right]
  $$

  - 噪声衡量了当前任务上任何模型所能达到的泛化误差下界

- 期望输出与真实标记的偏差：

  $$
  \mathrm{bias} \left( x \right) = \left( \bar{f} \left( x \right) - y \right) ^{2}
  $$

  - 偏差是所有训练集上平均预测值与真实值间的差异，衡量了模型本身的拟合能力

- 平均泛化误差：

  $$
  \mathbb{E} \left( f; \ D \right) = \mathbb{E}_{D} \left[ \left( f \left( x; \ D \right) - y_{D} \right)^{2} \right]
  $$

- 误差的偏差—方差分解：

  $$
  \mathbb{E} \left( f; \ D \right) = \mathrm{var} \left( x \right) + \mathrm{bias} \left( x \right) + \epsilon^{2}
  $$

  - 在训练不足时，模型拟合能力不强，更换训练集不足以使模型的拟合能力发生显著变化，此时偏差主导了泛化误差

  - 在训练充足后，模型拟合能力较强，训练集的轻微变动都会导致模型发生显著变化，此时方差主导了泛化误差

## 频率派 $\mathrm{VS}$ 贝叶斯派

### 频率派

- 概率表示反复实验时，事件发生的频率：

  - 抛硬币实验中，正面朝上的概率

- 频率派认为模型参数是固定的，经过无限次抽样后，可以得到模型的参数

- 进行参数估计时，假设参数固定但未知，目标是求解最可能的参数

  - 计算对数似然：

    $$
    L \left( \theta \right) = \sum_{i = 1}^{n} \log P \left( x_{i}|\theta \right)
    $$

  - 最大似然估计：

    $$
    \theta^{*}_{MLP} = \arg \max_{\theta} L \left( \theta \right)
    $$

### 贝叶斯派

- 概率表示对某个事件的确定程度：

  - 症状已知，判断某个病人患病的概率

- 贝叶斯派认为数据是固定的，在求解参数时考虑了参数的先验分布

- 进行参数估计时，假设参数服从某个分布，目标是求解该分布下最可能的参数值：

  - 计算参数的后验分布：

    $$
    P \left( \theta|X \right) = \frac{P \left( \theta, X \right)}{P \left( X \right)} = \frac{P \left( X|\theta \right) \cdot P \left( \theta \right)}{P \left( X \right)} \propto P \left( X|\theta \right) \cdot P \left( \theta \right) = P \left( \theta \right) \prod_{i = 1}^{n} P \left( x_{i}|\theta \right)
    $$

    $$
    L \left( \theta \right) = \log P \left( \theta \right) + \sum_{i = 1}^{n} \log P \left( x_{i}|\theta \right)
    $$

  - 最大后验估计：

    $$
    \theta^{*}_{MAP} = \arg \max_{\theta} L \left( \theta \right)
    $$

## 监督 $\mathrm{VS}$ 无监督 $\mathrm{VS}$ 半监督

### 监督

- 使用有标签数据进行模型训练

### 无监督

- 通过无标签数据进行模型训练

### 半监督

- 同时使用有标签、无标签数据进行模型训练

## 生成模型 $\mathrm{VS}$ 判别模型

### 生成模型

- 通过学习联合概率分布 $P \left( X, \ Y \right)$，然后求出 $P \left( Y|X \right)$ 作为预测模型：

  $$
  P \left( Y|X \right) = \frac{P \left( X, \ Y \right)}{P \left( X \right)}
  $$

- 生成模型可以还原出联合分布 $P \left(X, \ Y \right)$

- 当存在隐变量时，仍然可以使用生成学习方法

- 典型的生成模型包括：朴素贝叶斯、高斯混合模型、隐马尔可夫模型

### 判别模型

- 直接学习 $P \left( Y|X \right)$ 或决策函数 $f \left( X \right)$ 作为预测模型

- 判别模型直接面对预测，往往准确率更高

- 典型的判别模型包括：$\mathrm{kNN}$、决策树、逻辑回归、最大熵、$\mathrm{SVM}$、神经网络

## 分类 $\mathrm{VS}$ 标注 $\mathrm{VS}$ 回归

### 分类

- 从数据集中学习的分类模型或分类决策函数，称为分类器

- 通过分类器对新的输入进行输出的预测，称为分类

### 标注

- 输入是一个观测序列，输出是一个标记序列或状态序列

- 分类问题的推广，分类的输出是标量，标注的输出是向量

### 回归

- 用于预测输入变量到输出变量的映射关系

- 选择一条函数曲线，使其很好地拟合已知数据并很好地预测未知数据