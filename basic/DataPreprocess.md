<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 数据预处理

## 数据增强

- 由于直接采集新样本代价较高，通常对已有样本进行预处理，以扩充数据集大小

- 对于图像数据，一般采用如下方法：

	- 随机裁剪

		- 从原始图像上随机截取固定大小的图像作为新样本

	- 旋转与翻转

		- 旋转一定角度或水平翻转，改变图像内容朝向

	- 缩放

		- 按一定比例对图像进行放大或缩小

	- 添加噪声

		- 对整张图像加高斯噪声或椒盐噪声

	- 调整光照

		- 将 RGB 图像变换到 HSV 空间，固定 H，调整 SV

		- 关于颜色空间，参考 [ColorSpace.md](../vision/ColorSpace.md)

## 缺失值处理

### 直接删除

- 如果数据集较大，某些数据缺失值较多而数据量本身较少，直接删除这些缺失数据

### Mean / Median 估计

- 如果缺失值所占比例较小，采用该方法

- 广义插补

	- 计算该变量所有已知值的 Mean 或 Median，用来替换缺失值

- 相似插补

	- 根据某个属性计算 Mean 或 Median，用来替换相同属性的缺失值

		- 比如分别计算男、女的年龄，用来替换缺失值

### 热卡填充

- 在所有完整数据中，找到一个与缺失数据最相似的样本，将对应属性作为缺失值

- 概念简单，利用了数据间的关系；但问题在于相似的标准难以定义

### 预测模型

- 如果缺失值所占比例不大不小，采用该方法

- 创建预测模型来估计缺失值

- 将数据集分为两组：一组没有该变量的缺失值，另一组有缺失值

	- 在第一个组上训练，在第二组上预测

- 但是会存在以下问题：

	- 预测值通常比真实值更好

	- 如果缺失值与其他属性没有关系，模型的估计结果将变差

### kNN

- 对于缺失数据，计算 \\(k\\) 个最接近的预测样本，将这些样本的逆距离加权作为最终的填充值

- kNN 不需要为每个缺失属性创建预测模型，但是 \\(k\\) 值的选择不易确定

## 类别不平衡

### 扩大数据集

- 收集更多的数据，使正、负样本比例平衡

### 重采样

- 对大类的数据样本进行欠采样，以删除部分样本

- 对小类的数据样本进行过采样，以添加某些副本

### 合成样本

- 从小类所有样本每个属性的取值空间中随机选取一个值，组成新的样本

### 类别惩罚

- 如果分类任务是识别小类，可以增加小类样本的权重，降低大类样本的权重

### 衡量指标

- 除准确率外，考虑查准率、查全率、\\(F\\) 值、ROC 曲线等衡量指标

- 关于衡量指标，参考 [ModelEvaluation.md](ModelEvaluation.md)

## 特征选择

- 从 \\(N\\) 个特征中选择 \\(D\\) 个最好的特征

### 评价函数

#### 相关性

- 好的特征子集所包含的特征，应该与分类的相关度较高，而特征之间的相关度较低

- 不同属性之间的相关系数计算如下：

	$$
	R = \frac{cov(X, Y)}{\sqrt{var(X) \cdot var(Y)}}
	$$

#### 距离

- 好的特征子集所包含的特征，应该使得同类的样本距离尽可能小，不同类的样本距离尽可能大

#### 信息增益

- 信息增益越大，特征子集的分类效果越好

### 穷举式搜索

- 枚举所有特征组合，实用性不高

### 启发式搜索

#### 序列前向选择

- 特征子集 \\(X\\) 从空集开始，每次选择一个特征 \\(a\\)，使得评价函数最优

- 主要问题如下：

	- 只能加入特征，不能删除特征

	- 基于贪心算法，容易陷入局部最优值

#### 序列后向选择

- 特征子集 \\(X\\) 从全集开始，每次删除一个特征 \\(a\\)，使得特征函数最优

- 主要问题如下：

	- 只能删除特征，不能加入特征

	- 基于贪心算法，容易陷入局部最优值

#### 增 \\(L\\) 去 \\(R\\) 选择

- \\(L > R\\)

	- 特征子集 \\(X\\) 从空集开始，每轮先加入 \\(L\\) 个特征，再删除 \\(R\\) 个特征

- \\(R > L\\)

	- 特征子集 \\(X\\) 从全集开始，每轮先删除 \\(R\\) 个特征，再添加 \\(L\\) 个特征

- \\(L\\) 和 \\(R\\) 的选择是算法的关键

#### 序列浮动选择

- 与增 \\(L\\) 去 \\(R\\) 基本相同，不同点在于 \\(L, \ R\\) 会变化

- 结合了前向选择、后向选择、增 \\(L\\) 去 \\(R\\) 的特点，并弥补了它们的缺点

#### 决策树剪枝

- 生成一棵剪枝的决策树，未被剪掉的特征即为选中的特征子集

## 特征提取

- 将 \\(N\\) 维特征进行线性变化，选出 \\(D\\) 维作为最终特征

- 常见方法包括 PCA、LDA，参考 [DimensionReduction.md](DimensionReduction.md)