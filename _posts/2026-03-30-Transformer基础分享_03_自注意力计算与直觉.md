---
title: Transformer基础分享_03_自注意力计算与直觉
date: 2026-03-30
tags:
  - Transformer
  - SelfAttention
  - ScaledDotProductAttention
  - 深度学习入门
series: Transformer基础分享
status: 可发布
---

<!-- # Transformer 基础分享：自注意力计算与直觉 -->

> 本节回答四个问题：  
> 1) 三分天下之后，Q/K/V 如何相互作用？  
> 2) Softmax 权重到底在做什么？  
> 3) 为什么要除以 \\(\sqrt{d_k}\\)？  
> 4) 注意力为什么通常不对称？

---

## 1. 先看结论（表格）

| 问题 | 核心结论 | 公式/关键词 |
|---|---|---|
| 单个 token 怎么和其他 token 交互？ | 用该 token 的 \\(Q_i\\) 与所有 \\(K_j\\) 做点积打分 | \\(s_{ij}=Q_iK_j^\top\\) |
| 权重怎么得到？ | 对分数做 Softmax，得到和为 1 的注意力分布 | \\(\alpha_{ij}=\text{softmax}(s_{ij})\\) |
| 输出向量怎么得到？ | 用注意力权重对所有 \\(V_j\\) 加权求和 | \\(Z_i=\sum_j\alpha_{ij}V_j\\) |
| 为什么除 \\(\sqrt{d_k}\\)？ | 稳定数值尺度，避免 Softmax 过饱和 | \\(\frac{QK^\top}{\sqrt{d_k}}\\) |

---

## 2. 从“爱”这个字出发：一步一步算 \\(Z_2\\)

设句子为“我爱中国”，句长 \\(n=4\\)，维度 \\(d_k=5\\)。

每个字先“三分天下”，得到：

$$
Q_i, K_i, V_i \in \mathbb{R}^{1\times 5},\quad i\in\{1,2,3,4\}
$$

现在固定“爱”（第 2 个字），即用 \\(Q_2\\) 去和所有 \\(K_j\\) 交互：

$$
s_{21}=Q_2K_1^\top,\ s_{22}=Q_2K_2^\top,\ s_{23}=Q_2K_3^\top,\ s_{24}=Q_2K_4^\top
$$

其中每个 \\(s_{2j}\\) 都是标量。

运算前后的形状可以这样看：

- \\(Q_2\in\mathbb{R}^{1\times5}\\)
- \\(K_j\in\mathbb{R}^{1\times5}\\)，所以 \\(K_j^\top\in\mathbb{R}^{5\times1}\\)
- 因而 \\(Q_2K_j^\top\in\mathbb{R}^{1\times1}\\)（即标量）

### 2.0 维度流转速查表

| 步骤 | 表达式 | 形状 |
|---|---|---|
| 打分 | \\(s_{2j}=Q_2K_j^\top\\) | \\((1\times5)(5\times1)=1\times1\\) |
| 缩放 | \\(\tilde{s}_{2j}=s_{2j}/\sqrt{5}\\) | 标量 |
| 归一化 | \\(\alpha_{2j}=\text{softmax}(\tilde{s}_{2j})\\) | 标量 |
| 聚合 | \\(Z_2=\sum_{j=1}^{4}\alpha_{2j}V_j\\) | \\(1\times5\\) |

### 2.1 先缩放，再 softmax

\\(\alpha_{2j}\\) 表示“第 2 个 token（爱）对第 \\(j\\) 个 token 的注意力权重”，
它满足 \\(\alpha_{2j}\ge 0\\)

$$
\tilde{s}_{2j}=\frac{s_{2j}}{\sqrt{d_k}}=\frac{Q_2K_j^\top}{\sqrt{5}}
$$

$$
\alpha_{2j}=\frac{\exp\left(\frac{Q_2\cdot K_j}{\sqrt{5}}\right)}{\sum_{m=1}^{4}\exp\left(\frac{Q_2\cdot K_m}{\sqrt{5}}\right)}=\frac{e^{\tilde{s}_{2j}}}{\sum_{m=1}^{4}e^{\tilde{s}_{2m}}},\quad j=1,2,3,4
$$

所以 \\(\alpha_{21}+\alpha_{22}+\alpha_{23}+\alpha_{24}=1\\)。

若把第 2 行权重写成向量，可记为：

$$
\boldsymbol{\alpha}_2=[\alpha_{21},\alpha_{22},\alpha_{23},\alpha_{24}]\in\mathbb{R}^{1\times4}
$$

### 2.2 用权重加权所有 \\(V\\)

$$
Z_2=\alpha_{21}V_1+\alpha_{22}V_2+\alpha_{23}V_3+\alpha_{24}V_4
$$

由于每个 \\(V_j\in\mathbb{R}^{1\times5}\\) 且 \\(\alpha_{2j}\\) 是标量，所以：

$$
Z_2\in\mathbb{R}^{1\times5}
$$

> 直观理解：\\(Z_2\\) 是对全部 \\(V\\) 的“上下文加权摘要”，权重由“爱”对其他字的注意力决定。

---

## 3. 推广到整句：从 \\(X\\) 到 \\(Z\\)

同理可得 \\(Z_1,Z_2,Z_3,Z_4\\)，将它们按行拼接：

$$
Z=\begin{bmatrix}Z_1\\Z_2\\Z_3\\Z_4\end{bmatrix}\in\mathbb{R}^{4\times5}
$$

在这个例子中：

- 输入 \\(X\in\mathbb{R}^{4\times5}\\)
- 输出 \\(Z\in\mathbb{R}^{4\times5}\\)

因此，自注意力层在该设定下**不改变张量形状**。

对应到整句矩阵写法，还可以写成：

$$
A=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right),\quad A\in\mathbb{R}^{4\times4}
$$

$$
Z=AV,\quad (4\times4)(4\times5)=4\times5
$$

这里 \\(A\\) 的每一行就是一个 token 对整句 4 个 token 的注意力分布。

---

## 4. 为什么是内积？为什么还要缩放(\\(\frac{1}{\sqrt{d_k}}\\))？

### 4.1 内积的直觉

内积常用来衡量向量“对齐程度”。

设：

$$
A=(1,1),\ B=(1,1.1),\ C=(1,0.1)
$$

则：

$$
A\cdot B=2.1,\quad A\cdot C=1.1
$$

可见 \\(A\\) 与 \\(B\\) 更“接近”。

### 4.2 仅看内积大小, 不进行归一化的不足

若再取：

$$
D=(100,500),\quad A\cdot D=600
$$

内积数值会很大，但这主要受向量长度影响，不一定代表更好的语义匹配。

### 4.3 缩放点积的作用

Transformer 使用的是“缩放点积注意力”：

$$
\text{Attention}(Q,K,V)=\text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

除以 \\(\sqrt{d_k}\\) 的主要目的是稳定方差与数值范围，降低 Softmax 过饱和风险，让训练更稳定。

---

## 5. 为什么注意力不对称

一般来说，注意力是方向性的：

$$
\alpha_{i\to j}\neq\alpha_{j\to i}
$$

比如在“我爱中国”里，“中对国的注意力”通常不等于“国对中的注意力”。

在“杜鹃”中, 杜可以和"杜甫", "杜康"搭配, 但是鹃只有"杜鹃"唯一搭配, 因此杜对鹃注意力通常小于鹃对杜的注意力

- 在某个字的视角下，另一个字可能非常关键；
- 反过来则未必同样关键。

这就是注意力非对称。


---

## 6. SVG 硬编码图：Q/K/V 交互到 Z（分开画矩阵）

<svg width="100%" viewBox="0 0 1080 420" preserveAspectRatio="xMidYMid meet" style="max-width:1080px;height:auto;display:block;margin:12px auto;" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="QKV attention flow">
  <defs>
    <marker id="arrA" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#111827"/>
    </marker>
  </defs>

  <rect x="0" y="0" width="1080" height="420" fill="#fffef9"/>
  <text x="24" y="32" font-size="24" font-family="Georgia, serif" fill="#111827" font-weight="700">自注意力流程（示例：i=2）</text>

  <rect x="20" y="60" width="180" height="230" rx="10" fill="#e8f1ff" stroke="#355070" stroke-width="2"/>
  <text x="110" y="92" text-anchor="middle" font-size="22" font-family="Consolas, monospace" fill="#0f172a">X (4×5)</text>
  <text x="44" y="130" font-size="16" font-family="Consolas, monospace" fill="#0f172a">x1(我)</text>
  <text x="44" y="160" font-size="16" font-family="Consolas, monospace" fill="#0f172a">x2(爱) ← 当前查询</text>
  <text x="44" y="190" font-size="16" font-family="Consolas, monospace" fill="#0f172a">x3(中)</text>
  <text x="44" y="220" font-size="16" font-family="Consolas, monospace" fill="#0f172a">x4(国)</text>

  <line x1="210" y1="165" x2="320" y2="95" stroke="#111827" stroke-width="3" marker-end="url(#arrA)"/>
  <line x1="210" y1="165" x2="320" y2="175" stroke="#111827" stroke-width="3" marker-end="url(#arrA)"/>
  <line x1="210" y1="165" x2="320" y2="255" stroke="#111827" stroke-width="3" marker-end="url(#arrA)"/>
  <text x="230" y="88" font-size="15" font-family="Consolas, monospace" fill="#111827">×WQ</text>
  <text x="230" y="170" font-size="15" font-family="Consolas, monospace" fill="#111827">×WK</text>
  <text x="230" y="278" font-size="15" font-family="Consolas, monospace" fill="#111827">×WV</text>

  <rect x="340" y="20" width="180" height="120" rx="10" fill="#eafaf0" stroke="#2d6a4f" stroke-width="2"/>
  <text x="430" y="52" text-anchor="middle" font-size="28" font-family="Times New Roman, serif" fill="#0f172a">Q (4×5)</text>
  <text x="362" y="84" font-size="15" font-family="Consolas, monospace" fill="#0f172a">q1, q2, q3, q4</text>
  <text x="362" y="108" font-size="15" font-family="Consolas, monospace" fill="#0f172a">取 q2 参与打分</text>

  <rect x="340" y="150" width="180" height="120" rx="10" fill="#fff4e6" stroke="#9c6644" stroke-width="2"/>
  <text x="430" y="182" text-anchor="middle" font-size="28" font-family="Times New Roman, serif" fill="#0f172a">K (4×5)</text>
  <text x="362" y="214" font-size="15" font-family="Consolas, monospace" fill="#0f172a">k1, k2, k3, k4</text>
  <text x="362" y="238" font-size="15" font-family="Consolas, monospace" fill="#0f172a">与 q2 逐个内积</text>

  <rect x="340" y="280" width="180" height="120" rx="10" fill="#f3e8ff" stroke="#6d597a" stroke-width="2"/>
  <text x="430" y="312" text-anchor="middle" font-size="28" font-family="Times New Roman, serif" fill="#0f172a">V (4×5)</text>
  <text x="362" y="344" font-size="15" font-family="Consolas, monospace" fill="#0f172a">v1, v2, v3, v4</text>
  <text x="362" y="368" font-size="15" font-family="Consolas, monospace" fill="#0f172a">按权重加权求和</text>

  <line x1="530" y1="180" x2="650" y2="180" stroke="#111827" stroke-width="3" marker-end="url(#arrA)"/>
  <text x="544" y="168" font-size="14" font-family="Consolas, monospace" fill="#111827">s2j = q2·kj</text>

  <rect x="660" y="120" width="210" height="120" rx="10" fill="#fee2e2" stroke="#b91c1c" stroke-width="2"/>
  <text x="765" y="152" text-anchor="middle" font-size="20" font-family="Consolas, monospace" fill="#111827">scores s21..s24</text>
  <text x="765" y="180" text-anchor="middle" font-size="16" font-family="Consolas, monospace" fill="#111827">/ sqrt(5)</text>
  <text x="765" y="208" text-anchor="middle" font-size="16" font-family="Consolas, monospace" fill="#111827">softmax → α21..α24</text>

  <line x1="875" y1="180" x2="970" y2="180" stroke="#111827" stroke-width="3" marker-end="url(#arrA)"/>

  <rect x="980" y="145" width="90" height="70" rx="10" fill="#dcfce7" stroke="#15803d" stroke-width="2"/>
  <text x="1025" y="173" text-anchor="middle" font-size="18" font-family="Consolas, monospace" fill="#111827">Z2</text>
  <text x="1025" y="196" text-anchor="middle" font-size="14" font-family="Consolas, monospace" fill="#334155">1×5</text>

  <text x="650" y="275" font-size="15" font-family="Consolas, monospace" fill="#111827">Z2 = α21v1 + α22v2 + α23v3 + α24v4</text>
</svg>

---

## 7. 过渡到下一节

到这里，我们已经完整得到“单头注意力”的输出 \\(Z\\)。下一步通常是：

1. 多头并行（Multi-Head Attention）；
2. 拼接与线性变换；
3. 残差连接 + LayerNorm。

---

## 8. 一句话收束

**自注意力的本质是：用 Query 决定“看谁”，用 Key 计算“看多少”，用 Value 汇总“看到了什么”。**

---

## 系列导航

- 上一篇：[Transformer基础分享_02_位置编码与QKV](./2026-03-30-Transformer基础分享_02_位置编码与QKV.md)
- 下一篇：[Transformer基础分享_04_自注意力_矩阵视角](./2026-03-30-Transformer基础分享_04_自注意力_矩阵视角.md)
