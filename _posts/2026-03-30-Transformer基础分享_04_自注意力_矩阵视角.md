---
title: Transformer基础分享_04_自注意力_矩阵视角
date: 2026-03-30
tags:
  - Transformer
  - SelfAttention
  - 矩阵视角
  - 深度学习入门
series: Transformer基础分享
status: 可发布
---

# Transformer 基础分享：自注意力（矩阵视角）

> 这版专门把“向量逐个算”升级为“矩阵一次算完”，并尽量保留口语讲解里的辅助理解链条。

---

## 1. 一页速览（表格）

| 阶段 | 向量视角 | 矩阵视角 | 形状（本例） |
|---|---|---|---|
| 三分天下 | \(q_i=x_iW_Q\) 等 | \(Q=XW_Q\)，\(K=XW_K\)，\(V=XW_V\) | \(X,Q,K,V\in\mathbb{R}^{4\times5}\) |
| 打分 | \(s_{ij}=q_i\cdot k_j\) | \(S=QK^\top\) | \(S\in\mathbb{R}^{4\times4}\) |
| 归一化 | 对 \(s_{i1..4}\) 做 softmax | \(A=\text{softmax}(S/\sqrt{d_k})\) | \(A\in\mathbb{R}^{4\times4}\) |
| 聚合 | \(z_i=\sum_j\alpha_{ij}v_j\) | \(Z=AV\) | \(Z\in\mathbb{R}^{4\times5}\) |

---

## 2. 从输入矩阵开始：\(X\) 是什么

以“我爱中国”为例，按字切分后句长 \(n=4\)，嵌入维度 \(d=5\)，输入矩阵为：

$$
X=\begin{bmatrix}
 x_1 \\
 x_2 \\
 x_3 \\
 x_4
\end{bmatrix}\in\mathbb{R}^{4\times5}
$$

其中：

- \(x_1\) 对应“我”
- \(x_2\) 对应“爱”
- \(x_3\) 对应“中”
- \(x_4\) 对应“国”

这一步与口语版“每行一个字”完全一致。

---

## 3. 三分天下（矩阵版一次完成）

我们常说的逐行写法是：\(x_i\to q_i,k_i,v_i\)。

矩阵写法等价为：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

设 \(W_Q,W_K,W_V\in\mathbb{R}^{5\times5}\)，则：

$$
Q,K,V\in\mathbb{R}^{4\times5}
$$

这表示：

- \(Q\) 的第 \(i\) 行就是 \(q_i\)
- \(K\) 的第 \(i\) 行就是 \(k_i\)
- \(V\) 的第 \(i\) 行就是 \(v_i\)

---

## 4. 相互作用：为什么是 \(QK^\top\)

向量角度是“每个 \(q_i\) 和所有 \(k_j\) 做内积”。

矩阵角度直接写成：

$$
S=QK^\top
$$

其中 \(S\) 是未归一化的注意力打分矩阵（score matrix）。

形状：

$$
(4\times5)(5\times4)=4\times4
$$

元素解释：

$$
S_{ij}=q_i\cdot k_j
$$

例如固定 \(i=2\)（对应“爱”），可展开为：

$$
[s_{21},s_{22},s_{23},s_{24}]=
[q_2\cdot k_1,\ q_2\cdot k_2,\ q_2\cdot k_3,\ q_2\cdot k_4]
$$

这就是我们常说的“\(Q_2\) 依次与所有 \(K_j\) 做内积打分”。

---

## 5. 缩放与 Softmax

注意力不是直接用 \(S\)，而是：

$$
A=\operatorname{softmax}\left(\frac{S}{\sqrt{d_k}}\right)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
$$

本例 \(d_k=5\)，所以除以 \(\sqrt{5}\)。

第 \(i\) 行 softmax 后得到权重向量：

$$
\boldsymbol{\alpha}_i=[\alpha_{i1},\alpha_{i2},\alpha_{i3},\alpha_{i4}],
\quad
\sum_{j=1}^{4}\alpha_{ij}=1
$$

固定 \(i=2\) 时，写成展开式就是：

$$
\alpha_{2j}=\frac{\exp\left(s_{2j}/\sqrt{d_k}\right)}{\sum_{m=1}^{4}\exp\left(s_{2m}/\sqrt{d_k}\right)}
\quad (j=1,2,3,4)
$$

---

## 6. 加权求和得到输出 \(Z\)

向量角度（以“爱”为例）是：

$$
z_2=\alpha_{21}v_1+\alpha_{22}v_2+\alpha_{23}v_3+\alpha_{24}v_4
$$

一般式可写成：

$$
z_i=\sum_{j=1}^{4}\alpha_{ij}v_j
=\alpha_{i1}v_1+\alpha_{i2}v_2+\cdots+\alpha_{i4}v_4
$$

矩阵角度统一写成：

$$
Z=AV
$$

---

## 系列导航

- 上一篇：[Transformer基础分享_03_自注意力计算与直觉](./2026-03-30-Transformer基础分享_03_自注意力计算与直觉.md)

形状检查：

$$
(4\times4)(4\times5)=4\times5
$$

所以自注意力层在这个设定下：

- 输入 \(X\) 是 \(4\times5\)
- 输出 \(Z\) 也是 \(4\times5\)

**形状不变，但语义增强**：每个 \(z_i\) 都融合了全句其他 token 的信息。

---

## 7. 口语直觉保留：为什么叫“自注意力”

“自注意力”的“自”，是指同一句子内部元素相互看彼此。

- \(z_1\) 仍对应“我”，但已经融合了“爱/中/国”的信息。
- \(z_2\) 仍对应“爱”，但权重由“爱对全句的关注分布”决定。

这就是我们常说的“经过三分天下与交互后，每个字都带上了全句上下文信息”。

---

## 8. 语境重心例子（保留并书面化）

“我爱中国”在不同语境中重心可能不同：

- 若语境强调对象，注意力可能更集中在“中国”；
- 若语境强调动作，可能更关注“爱”。

这也是后续“多头自注意力”有价值的原因：
不同头可以关注不同关系子空间，而不是只提取单一重心。

## 9. 总结

从 \(X\) 到 \(Z\) 的完整公式链如下：

$$
Q=XW_Q,\quad K=XW_K,\quad V=XW_V
$$

$$
S=QK^\top
$$

$$
A=\operatorname{softmax}\left(\frac{S}{\sqrt{d_k}}\right)
=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)
$$

$$
Z=AV
$$
