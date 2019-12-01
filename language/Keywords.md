# 关键字

## $\mathrm{tf_{t, \ d}}$

- 词项频率（$\mathrm{term \ frequency}$）

- 词项 $t$ 在文档 $d$ 中的出现次数

- $\mathrm{tf}$ 越大，词项越重要

## $\mathrm{df_{t}}$

- 文档频率（$\mathrm{document \ frequency}$）

- 出现词项 $t$ 的所有文档数

- $\mathrm{df}$ 越大，词项越不重要

## $\mathrm{idf_{t}}$

- 逆文档频率（$\mathrm{inverse \ document \ frequency}$）

- 与 $\mathrm{df_{t}}$ 负相关：$\mathrm{idf_{t}} = \lg \frac {N} {\mathrm{df_{t}}}$

- $\mathrm{idf}$ 越大，词项越重要

## $\mathrm{tf-idf_{t, \ d}}$

- $\mathrm{tf-idf_{t, \ d}} = \mathrm{tf_{t, \ d}} \cdot \mathrm{idf_{t}}$

- 当词项 $t$ 在少数几篇文档中多次出现时，权重最大