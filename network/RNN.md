<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# Recurrent Neural Networks

## 背景介绍

- 在自然语言处理中，序列的当前输出与之前的输出有关：

	- 即 RNN 记忆到了目前为止已经计算过的信息

## 数学原理

![img](images/rnn.png)

- \\(x\_{t}\\) 是 \\(t\\) 时刻的输入：

	- 比如某个词的 one-hot 编码向量

- \\(s\_{t}\\) 是 \\(t\\) 时刻的隐状态，通过当前时刻的输入和前一时刻的隐状态得到：

	$$ s\_{t} = f \left( Ux\_{t} + Ws\_{t-1} \right) $$
	
	- 其中 \\(f\\) 为激活函数：ReLU 或 Tanh

	- \\(s\_{t}\\) 可以捕获到之前所有时刻产生的信息

- \\(o\_{t}\\) 是 \\(t\\) 时刻的输出：

	- 如果要预测句子的下一个词，那么 \\(o\_{t}\\) 就是包含所有词的概率向量：

		$$ o\_{t} = softmax \left( Vs\_{t} \right) $$
	
	- \\(o\_{t}\\) 只依赖于 \\(t\\) 时刻的隐状态

- RNN 中所有时刻共享相同的参数 \\(U, \ V, W\\)，减少了需要学习的参数量