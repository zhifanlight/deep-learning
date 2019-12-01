# 矩阵基础

## 正定阵

- 一个实对称矩阵 $M$ 是正定的，当且仅当对所有非零实向量 $z$，都有 $z^{T} M z > 0$

- 正定阵的特征值都是正数

## 半正定阵

- 一个实对称矩阵 $M$ 是半正定的，当且仅当对所有非零实向量 $z$，都有 $z^{T} M z \geq 0$

- 半正定阵的特征值都非负

## 方阵的迹

- 主对角线上元素之和

- 特征值之和

### 迹的性质

#### 性质 $1$：转置不变

$$
\mathrm{tr} \left( A \right) = \mathrm{tr} \left( A^{T} \right)
$$

#### 性质 $2$：循环不变

$$
\mathrm{tr} \left( AB \right) = \mathrm{tr} \left( BA \right), \quad \mathrm{tr} \left( ABC \right) = \mathrm{tr} \left( BCA \right) = \mathrm{tr} \left( CAB \right), \quad \cdots
$$

- 设 $A$ 为 $M \times N$ 矩阵，$B$ 为 $N \times M$ 矩阵：

  $$
  \mathrm{tr} \left( AB \right) = \sum_{i = 1}^{M} \left( AB \right)_{ii} = \sum_{i = 1}^{M} \sum_{j = 1}^{N} a_{ij} b_{ji} = \sum_{j = 1}^{N} \sum_{i = 1}^{M} b_{ji}a_{ij} = \sum_{j = 1}^{N} \left( BA \right)_{jj} = \mathrm{tr} \left( BA \right)
  $$

- 由 $\mathrm{tr} \left( AB \right) = \mathrm{tr} \left( BA \right)$ 可得：$\mathrm{tr} \left( ABC \right) = \mathrm{tr} \left( BCA \right) = \mathrm{tr} \left( CAB \right)$

#### 性质 $3$：线性可加

$$
\mathrm{tr} \left( \alpha \cdot A + \beta \cdot B \right) = \alpha \cdot \mathrm{tr} \left( A \right) + \beta \cdot \mathrm{tr} \left( B \right)
$$

- 其中，$\alpha, \ \beta$ 均为标量，$A, \ B$ 为同维度方阵

## 方阵特征值、特征向量

- 方阵 $A$ 的特征值 $\lambda$、特征向量 $v$ 满足以下关系：

  $$
  Av = \lambda v \quad \Leftrightarrow \quad \left( A - \lambda I \right) v = 0
  $$

- 令 $\mathrm{det} \left( A - \lambda I \right) = 0$ 可计算特征值，将特征值代入上式可计算特征向量

- 迹等于特征值之和，行列式等于特征值之积：

  - 假设 $ A = \left[ \begin{matrix} a & b \\ c & d \end{matrix} \right], \ v = \left[ \begin{matrix} m \\ n \end{matrix} \right] $，代入上式并化简可得：

    $$
    \lambda^{2} - \left( a + d \right) \lambda + \left( ad - bc \right) = 0
    $$

  - 求解上述一元二次方程可得：

    $$
    \left\{ \begin{matrix} \lambda_{1} + \lambda_{2} = a + d = \mathrm{tr} \left( A \right) \\ \lambda_{1} \cdot \lambda_{2} = ad - bc = \mathrm{det} \left( A \right) \end{matrix} \right.
    $$

- 如果向量 $x$ 是矩阵 $A$ 的特征向量，那么向量 $\beta x$ 也是其特征向量

  - 因此，通常说的特征向量是指单位特征向量

## 矩阵的秩

- 矩阵中最大的线性无关向量个数：

  - 行秩：矩阵中最大的线性无关行向量个数

  - 列秩：矩阵中最大的线性无关列向量个数

  - 秩：矩阵的秩等于其行秩，也等于其列秩

- 对于 $M \times N$ 的矩阵 $A$，当 $R \left( A \right) = \min \{M, \ N\}$ 时，矩阵 $A$ 为满秩矩阵

### 秩与非零解

- 判断 $A v = 0$ 是否有解，可以转化 $A$ 的列向量能否通过线性组合成为零向量

- 对于 $M \times N$ 的矩阵，当 $M < N$ 时一定有非零解：

  - 以 $2 \times 3$ 矩阵 $A = \left[ \begin{matrix} a & b & c \\ d & e & f \end{matrix} \right]$ 为例，此时 $v$ 可表示为 $\left[ \begin{matrix} x \\ y \\ z \end{matrix} \right]$

  - 代入矩阵可得：

    $$
    \left\{ \begin{matrix} ax + by + cz = 0 \\ dx + ey + fz = 0 \end{matrix} \right.
    $$

  - 整理上式可得：

    $$
    \left[ \begin{matrix} a \\ d \end{matrix} \right] x + \left[ \begin{matrix} b \\ e \end{matrix} \right] y + \left[ \begin{matrix} c \\ f \end{matrix} \right] z = \left[ \begin{matrix} 0 \\ 0 \end{matrix} \right]
    $$

  - 由于 $A$ 的秩不超过 $2$，上述 $3$ 个列向量必然线性相关，即存在非零解

- 对于 $N \times N$ 的方阵 $A$：

  - 当 $A$ 满秩时，不存在非零解

  - 当 $A$ 不满秩时，存在非零解

## 矩阵求导

- 若存在矩阵 $H$ 使下式成立，则 $\nabla_{X} f \left( X \right) = H$，

  $$
  \lim_{t \rightarrow 0} \frac{f \left( X + t \cdot W \right) - f \left( X \right)}{t} = \mathrm{tr} \left( W^{T} H \right)
  $$

  - 其中 $W$ 表示与 $X$ 维度相同的任意矩阵

- 由于机器学习的输出通常为标量，因此可以假设 $f \left( X \right) = \mathrm{tr} \left( X \right)$，然后进行求导

## 向量求导

- 假设 $x = \left[ \begin{matrix} x_{1} \\ x_{2} \\ x_{3} \end{matrix} \right], \ y = \left[ \begin{matrix} y_{1} \\ y_{2} \end{matrix} \right]$ 都是列向量，$z$ 是标量

- 标量与向量、向量与向量的求导结果都存在两种互为转置的形式

### 分子布局

- 对分母向量转置，结果与分子保持一致

  - 标量对列向量求导得到行向量：

    $$
    \frac{\partial z}{\partial x} = \left[ \begin{matrix} \frac{\partial z}{\partial x_{1}} & \frac{\partial z}{\partial x_{2}} & \frac{\partial z}{\partial x_{3}} \end{matrix} \right]
    $$

  - 列向量对标量求导得到列向量：

    $$
    \frac{\partial x}{\partial z} = \left[ \begin{matrix} \frac{\partial x_{1}}{\partial z} \\ \frac{\partial x_{2}}{\partial z} \\ \frac{\partial x_{3}}{\partial z} \end{matrix} \right]
    $$

  - 列向量对列向量求导，分子按列向量布局，分母按行向量布局：

    $$
    \frac{\partial y}{\partial x} = \left[ \begin{matrix} \frac{\partial y_{1}}{\partial x_{1}} & \frac{\partial y_{1}}{\partial x_{2}} & \frac{\partial y_{1}}{\partial x_{3}} \\ \frac{\partial y_{2}}{\partial x_{1}} & \frac{\partial y_{2}}{\partial x_{2}} & \frac{\partial y_{2}}{\partial x_{3}} \end{matrix} \right]
    $$

### 分母布局

- 对分子向量转置，结果与分母保持一致

  - 标量对列向量求导得到列向量：

    $$
    \frac{\partial z}{\partial x} = \left[ \begin{matrix} \frac{\partial z}{\partial x_{1}} \\ \frac{\partial z}{\partial x_{2}} \\ \frac{\partial z}{\partial x_{3}} \end{matrix} \right]
    $$

  - 列向量对标量求导得到行向量：

    $$
    \frac{\partial x}{\partial z} = \left[ \begin{matrix} \frac{\partial x_{1}}{\partial z} & \frac{\partial x_{2}}{\partial z} & \frac{\partial x_{3}}{\partial z} \end{matrix} \right]
    $$

  - 列向量对列向量求导，分子按行向量布局，分母按列向量布局：

    $$
    \frac{\partial y}{\partial x} = \left[ \begin{matrix} \frac{\partial y_{1}}{\partial x_{1}} & \frac{\partial y_{2}}{\partial x_{1}} \\ \frac{\partial y_{1}}{\partial x_{2}} & \frac{\partial y_{2}}{\partial x_{2}} \\ \frac{\partial y_{1}}{\partial x_{3}} & \frac{\partial y_{2}}{\partial x_{3}} \end{matrix} \right]
    $$

## 向量正交化

- 将一组线性无关的 $N$ 维向量 $\alpha_{1}, \ \alpha_{2}, \ \cdots, \ \alpha_{M}$ 转换为一组正交向量 $\beta_{1}, \ \beta_{2}, \ \cdots, \ \beta_{M}$

- 若将上述正交向量分别单位化，可以得到 $N$ 维空间中的一组正交基 $\eta_{1}, \ \eta_{2}, \ \cdots, \ \eta_{M}$

### 施密特正交化

#### 正交向量

- 选定第一个正交向量：

  $$
  \beta_{1} = \alpha_{1}
  $$

- 计算第二个正交向量：

  - 设 $\beta_{2}$ 具有如下形式：

    $$
    \beta_{2} = \alpha_{2} - k \beta_{1}
    $$

  - 由正交关系可得：

    $$
    \left \langle \beta_{2}, \ \beta_{1} \right \rangle = \left \langle \alpha_{2}, \ \beta_{1} \right \rangle - k \left \langle \beta_{1}, \ \beta_{1} \right \rangle = 0
    $$

  - 由 $\left \langle \beta_{1}, \ \beta_{1} \right \rangle > 0$ 可得：

    $$
    k = \frac{\left \langle \alpha_{2}, \ \beta_{1} \right \rangle}{\left \langle \beta_{1}, \ \beta_{1} \right \rangle}
    $$

  - 代入 $\beta_{2}$ 得第二个正交向量：

    $$
    \beta_{2} = \alpha_{2} - \frac{\left \langle \alpha_{2}, \ \beta_{1} \right \rangle}{\left \langle \beta_{1}, \ \beta_{1} \right \rangle} \beta_{1}
    $$

- 计算第三个正交向量：

  - 设 $\beta_{3}$ 具有如下形式：

    $$
    \beta_{3} = \alpha_{3} - k_{1} \beta_{1} - k_{2} \beta_{2}
    $$

  - 由正交关系可得：

    $$
    \left \langle \alpha_{3}, \ \beta_{1} \right \rangle - k_{1} \left \langle \beta_{1}, \ \beta_{1} \right \rangle = 0, \quad \left \langle \alpha_{3}, \ \beta_{2} \right \rangle - k_{2} \left \langle \beta_{2}, \ \beta_{2} \right \rangle = 0
    $$

  - 由 $\left \langle \beta_{1}, \ \beta_{1} \right \rangle > 0, \ \left \langle \beta_{2}, \ \beta_{2} \right \rangle > 0$ 可得：

    $$
    k_{1} = \frac{\left \langle \alpha_{3}, \ \beta_{1} \right \rangle}{\left \langle \beta_{1}, \ \beta_{1} \right \rangle}, \quad k_{2} = \frac{\left \langle \alpha_{3}, \ \beta_{2} \right \rangle}{\left \langle \beta_{2}, \ \beta_{2} \right \rangle}
    $$

  - 代入 $\beta_{3}$ 得第三个正交向量：

    $$
    \beta_{3} = \alpha_{3} - \sum_{k = 1}^{2} \frac{\left \langle \alpha_{3}, \ \beta_{k} \right \rangle}{\left \langle \beta_{k}, \ \beta_{k} \right \rangle} \beta_{k}
    $$

- 计算第 $j$ 个正交向量：

  $$
  \beta_{j} = \alpha_{j} - \sum_{k = 1}^{j - 1} \frac{ \left \langle \alpha_{j}, \ \beta_{k} \right \rangle}{\left \langle \beta_{k}, \ \beta_{k} \right \rangle} \beta_{k}
  $$

#### 正交基

- 对 $\beta_{j}$ 单位化得正交基：

  $$
  \eta_{j} = \frac{\beta_{j}}{||\beta_{j}||_{2}}
  $$

## 正交矩阵

- 行向量和列向量分别标准正交的矩阵：

  $$
  AA^{T} = A^{T}A = I
  $$

- 正交矩阵求逆代价较小：

  $$
  A^{-1} = A^{T}
  $$

## 实对称矩阵对角化

- 对于 $N \times N$ 的实对称矩阵 $A$，存在正交矩阵 $Q$ 使得：

  $$
  Q^{-1}AQ = \left[ \begin{matrix} \lambda_{1} & & &  \\ & \lambda_{2} & & \\ & & \ddots & \\ & & & \lambda_{N} \end{matrix} \right]
  $$

  - 其中，$\lambda_{i}$ 是 $A$ 的第 $i$ 个特征值，$Q$ 的每一列是 $A$ 单位正交化的特征向量

- 由于 $QQ^{T} = I$，对角化结果可进一步表示为：

  $$
  Q^{T}AQ = \left[ \begin{matrix} \lambda_{1} & & &  \\ & \lambda_{2} & & \\ & & \ddots & \\ & & & \lambda_{N} \end{matrix} \right]
  $$

## 雅各比矩阵

- 假设 $f$ 将 $N$ 维向量 $x$ 映射成 $M$ 维向量 $y$，则雅各比矩阵定义如下：

  $$
  J = \left[ \begin{matrix} \frac{\partial{y_{1}}}{\partial{x_{1}}} & \frac{\partial{y_{1}}}{\partial{x_{2}}} & \cdots &  \frac{\partial{y_{1}}}{\partial{x_{N}}} \\ \frac{\partial{y_{2}}}{\partial{x_{1}}} & \frac{\partial{y_{2}}}{\partial{x_{2}}} & \cdots &  \frac{\partial{y_{2}}}{\partial{x_{N}}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial{y_{M}}}{\partial{x_{1}}} & \frac{\partial{y_{M}}}{\partial{x_{2}}} & \cdots & \frac{\partial{y_{M}}}{\partial{x_{N}}} \end{matrix} \right]
  $$

- 相当于按分子布局进行列向量求导 $\frac{\partial{y}}{\partial{x}}$

## 海森矩阵

- 假设 $f$ 将 $N$ 维向量 $x$ 映射成标量，则海森矩阵定义如下：

  $$
  J = \left[ \begin{matrix} \frac{\partial^{2}{f}}{\partial{x_{1}^{2}}} & \frac{\partial^{2}{f}}{\partial{x_{1}} \partial{x_{2}}} & \cdots &  \frac{\partial^{2}{f}}{\partial{x_{1}} \partial{x_{N}}} \\ \frac{\partial^{2}{f}}{\partial{x_{1}} \partial{x_{2}}} & \frac{\partial^{2}{f}}{\partial{x_{2}^{2}}} & \cdots & \frac{\partial^{2}{f}}{\partial{x_{2}} \partial{x_{N}}} \\ \vdots & \vdots & \ddots & \vdots \\ \frac{\partial^{2}{f}}{\partial{x_{N}} \partial{x_{1}}} & \frac{\partial^{2}{f}}{\partial{x_{N}} \partial{x_{2}}} & \cdots & \frac{\partial^{2}{f}}{\partial{x_{N}^{2}}} \end{matrix} \right]
  $$