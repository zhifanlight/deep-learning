<script type="text/javascript" src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=default"></script>

# 关键字

## \\(tf_{t,d}\\)

- 词项频率（term frequency）

- 词项 \\(t\\) 在文档 \\(d\\) 中的出现次数

- \\(tf\\) 越大，词项越重要

## \\(df_{t}\\)

- 文档频率（document frequency）

- 出现词项 \\(t\\) 的所有文档数

- \\(df\\) 越大，词项越不重要

## \\(idf_{t}\\)

- 逆文档频率（inverse document frequency）

- 与 \\( df\_{t} \\) 负相关：\\(idf_{t} = lg \frac {N} {df\_{t}}\\)

- \\(idf\\) 越大，词项越重要

## \\(tf\\)-\\(idf_{t,d}\\)

- \\(tf\\)-\\(idf\_{t,d} = tf\_{t,d} \cdot idf_{t}\\)

- 当词项 \\(t\\) 在少数几篇文档中多次出现时，权重最大