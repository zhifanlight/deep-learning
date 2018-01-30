<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 决策树

## 背景介绍

- 决策树是一个树结构：每个非叶节点表示一个特征的测试，每个分支代表这个特征的取值，每个叶节点代表一个类别

- 使用决策树进行决策时，从根节点开始，根据待分类的特征选择相应分支，直到到达叶节点，将叶节点的类别作为决策结果

## 决策树

### ID3

#### 基本思想

- 计算当前子集中每个属性的信息增益，选择信息增益最大的属性进行分裂

	- 信息增益（互信息）描述了样本类别对该属性的依赖程度

		- 关于互信息，参考 [Entropy.md](Entropy.md)

#### 算法流程

- 初始化时，当前集合 \\(D\\) 是整个训练集

- 计算当前集合 \\(D\\) 的信息熵：

	$$ Ent(D) = - \sum\_{k=1}^{K} p\_{k} \cdot log\_{2} \ {p\_{k}} $$

	- 其中 \\(K\\) 是 \\(D\\) 中的类别数，\\(p\_{k}\\) 是每个类别的比例

- 计算当前集合 \\(D\\) 中每个未处理属性 \\(a\\) 的信息增益：

	$$ Gain(D,a) = Ent(D) - \sum\_{v \in V} \frac{|D\_{v}|}{|D|} Ent(D\_{v}) $$

	- 其中 \\(V\\) 是 \\(D\\) 中 \\(a\\) 属性的取值集合，\\(D\_{v}\\) 是 \\(D\\) 中 \\(a=v\\) 的样本集合

- 选择最大信息增益 \\(G\_{max}\\) 对应的属性，将 \\(D\\) 划分成不同子集 \\(D\_{1}, D\_{2}, \cdots, D\_{|V|}\\)

- 重复上述三步，直到 \\(D\_{i}\\) 满足以下任一条件：

	- \\(D\_{i}\\) 中的样本已经属于同一类别，此时直接将该类别标签作为叶节点，停止划分

	- \\(G\_{max}\\) 小于阈值 \\(\epsilon\\)，此时直接将 \\(D\_{i}\\) 所有样本划分为同一类别，并将大多数样本对应的类别标签作为叶节点，停止划分

	- 处理完所有属性后 \\(D\_{i}\\) 中的样本还不属于同一类别，此时将 \\(D\_{i}\\) 中大多数样本对应的类别标签作为叶节点，停止划分

#### 算法分析

- 每次分裂时，倾向于选择取值数目多的属性，而不是最好的属性

	- 极端情况下，比如计算 ID 属性对应的信息增益时，由于每个 \\(D\_{v}\\) 只有一个样本，因此每个 \\(Ent(D\_{v})=0\\)，此时信息增益最大，然而对分类无益

### C4.5

#### 基本思想

- 计算当前子集中每个属性的增益率，选择增益率最大的属性进行分裂

	- 增益率越大，则信息增益越大，同时类别较少

#### 算法流程

- 与 ID3 基本相同，区别在于划分节点是最大增益率对应的属性

- 增益率计算如下：

	$$ Ratio(D,a) = \frac{Gain(D,a)}{Split(a)} $$

	- 其中 \\(Split(a)\\) 是属性 \\(a\\) 的分裂熵：

		$$ Split(a) = - \sum\_{v \in V} \frac{|D\_{v}|}{|D|} log\_{2} \frac{|D\_{v}|}{|D|} $$

		- 类别越多越混乱，分裂熵越大，增益率越小

#### 算法分析

- 解决了 ID3 选择分裂属性的问题

- 但每次分裂时，倾向于选择取值数目少的属性

### CART

#### 基本思想

- 计算当前子集中每个属性的基尼指数，选择基尼指数最小的属性进行分裂

	- 基尼指数衡量样本集的类别不纯净度，基尼指数越小，样本集类别越纯净

#### 算法流程

- 初始化时，当前集合 \\(D\\) 是整个训练集

- 计算当前集合 \\(D\\) 的基尼值：

	$$ Gini(D) = 1 - \sum\_{k=1}^{K} p\_{k}^{2} $$

	- 其中 \\(K\\) 是 \\(D\\) 中的类别数，\\(p\_{k}\\) 是每个类别的比例

	- 当所有样本属于一类时，基尼值为 \\(0\\)；类别越多，基尼值越大

- 计算当前集合 \\(D\\) 中每个未处理属性 \\(a\\) 的基尼指数：

	$$ Index(D,a) = \frac{|D\_{y}|}{|D|} Gini(D\_{y}) + \frac{|D\_{n}|}{|D|} Gini(D\_{n}) $$

	- 其中 \\(D\_{y}\\) 是 \\(D\\) 中 \\(a = v\\) 的样本集合，\\(D\_{n}\\) 是 \\(D\\) 中 \\(a \neq v\\) 的样本集合

- 选择最小基尼指数 \\(I\_{min}\\) 对应的属性，将 \\(D\\) 划分成两个子集 \\(D\_{1}, D\_{2}\\)

- 重复上述三步，直到 \\(D\_{i}\\) 满足以下任一条件：

	- \\(D\_{i}\\) 中的样本已经属于同一类别，此时直接将该类别标签作为叶节点，停

	- \\(I\_{min}\\) 小于阈值 \\(\epsilon\\)，此时基本基本属于同一类，将大多数样本对应的类别标签作为叶节点，停止划分

	- \\(D\_{i}\\) 的样本数小于阈值 \\(T\\)，此时继续划分的意义不大，将大多数样本对应的类别标签作为叶节点，停止划分

#### 算法分析

- 分类效果较好

- 无论属性取值数目多少，每次仅把当前集合分为两个子集

### 决策树剪枝

- 为了防止过拟合，需要对决策树进行剪枝处理

#### 预剪枝

- 在生成决策树的过程中进行剪枝，可能会导致欠拟合

- 从根开始依次处理每一个节点，对于当前选中的待分裂属性：

	- 如果划分后可以提升泛化性能，则进行划分；否则直接把当前节点置为叶节点，类别标签为当前集合中样本数最多的类别

#### 后剪枝

- 先生成决策树再进行剪枝，时间开销较大

- 从底层每一个非叶节点开始：

	- 如果合并其子树可以提升泛化性，则将该子树替换为叶节点，类别标签为当前集合中样本数最多的类别；否则保留子树，不进行替换