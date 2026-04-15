---
title: Transformer基础分享_06_编码器与LayerNorm
date: 2026-04-02
tags:
  - Transformer
  - Encoder
  - LayerNorm
  - 深度学习入门
status: 可发布
---

<!-- # Transformer 基础分享：编码器与 LayerNorm（发布版） -->

> 本节目标：
> 1) 讲清 BN 与 LN 的统计维度差异；
> 2) 解释为什么 NLP 中 LN 更常见；
> 3) 用矩阵视角梳理一个 Encoder Block 到整个 Encoder 的结构。

---

## 1. 结论先看（表格）

| 问题 | 核心结论 | 关键词 |
|---|---|---|
| BN 和 LN 的区别？ | 统计维度不同 | BN 跨 batch/通道，LN 常沿特征维 |
| 文本里为什么偏向 LN？ | 变长序列 + padding 干扰 BN 统计 | 动态长度 |
| Encoder Block 输入输出 shape？ | 通常保持一致 | 残差连接（Add） |
| Encoder 最终输出是什么？ | memory（编码信息矩阵） | 给 Decoder 使用 |

---

## 2. BN 与 LN：从“统计谁”开始

设图像张量（N,H,W,C）示意为：

$$
X\in\mathbb{R}^{10\times28\times28\times3}
$$

- \\(10\\)：batch size
- \\(28,28\\)：空间尺寸
- \\(3\\)：通道数

### 2.1 Batch Normalization（BN）

BN 的直觉是“按通道统计”。
例如第 1 个通道的均值可写为：

$$
\mu_{c=1}=\frac{1}{10\cdot28\cdot28}\sum_{n=1}^{10}\sum_{h=1}^{28}\sum_{w=1}^{28}X_{n,h,w,1}
$$

方差同理。

### 2.2 Layer Normalization（LN）

在 NLP 常见张量中，设：

$$
X\in\mathbb{R}^{B\times T\times D}
$$

- \\(B\\)：batch size
- \\(T\\)：序列长度
- \\(D\\)：嵌入维度

LN 的常见做法是：对每个 token 的最后一维 \\(D\\) 单独归一化。

对某个样本第 \\(t\\) 个 token：

$$
\mu_{b,t}=\frac{1}{D}\sum_{i=1}^{D}X_{b,t,i},
\quad
\sigma^2_{b,t}=\frac{1}{D}\sum_{i=1}^{D}(X_{b,t,i}-\mu_{b,t})^2
$$

$$
\mathrm{LN}(X_{b,t,i})=\gamma_i\frac{X_{b,t,i}-\mu_{b,t}}{\sqrt{\sigma^2_{b,t}+\varepsilon}}+\beta_i
$$

> **辅助理解举例（数值演算版）：**
> 
> 假设有一个输入张量 \\(X \in \mathbb{R}^{2\times3}\\)，表示 2 个 token，每个 token 3 维特征：
> 
> $$
> X = \begin{bmatrix}
> 1 & 2 & 3 \\
> 4 & 5 & 6
> \end{bmatrix}
> $$
> 
> 对第 1 个 token（第 1 行）做 LayerNorm：
> 
> - 均值 \\(\mu_1 = (1+2+3)/3 = 2\\)
> - 方差 \\(\sigma_1^2 = [(1-2)^2 + (2-2)^2 + (3-2)^2]/3 = (1+0+1)/3 = 0.666...\\)
> - 归一化后：
>   - \\(z_1 = (1-2)/\sqrt{0.666...+\varepsilon}\\)
>   - \\(z_2 = (2-2)/\sqrt{0.666...+\varepsilon}\\)
>   - \\(z_3 = (3-2)/\sqrt{0.666...+\varepsilon}\\)
> 
> 对第 2 个 token（第 2 行）同理：
> 
> - 均值 \\(\mu_2 = (4+5+6)/3 = 5\\)
> - 方差 \\(\sigma_2^2 = [(4-5)^2 + (5-5)^2 + (6-5)^2]/3 = (1+0+1)/3 = 0.666...\\)
> - 归一化后：
>   - \\(z_4 = (4-5)/\sqrt{0.666...+\varepsilon}\\)
>   - \\(z_5 = (5-5)/\sqrt{0.666...+\varepsilon}\\)
>   - \\(z_6 = (6-5)/\sqrt{0.666...+\varepsilon}\\)
> 
> 这样，每个 token 的 3 维特征都被单独归一化，均值为 0，方差为 1, 这种归一化又被称为Z-score归一

---

## 3. 为什么文本里 LN 更常见

文本是动态长度输入，不同样本 token 数不一致。

即使通过 padding 对齐长度，padding 位的统计意义也较弱。
在 BN 场景下，跨样本统计更容易受这种无效位置影响。

> **辅助理解举例：**
> 
> 假设一个 batch 里有 5 句话：
> 
> - 我爱中国
> - 我爱吃饭
> - 我爱洗澡
> - 我很高大
> - 我喜欢在太阳下吃葡萄干
> 
> 前 4 个样本只有 4 个词，第 5 个样本有十几个词。若按 BN 统计，每个位置的均值/方差都要跨样本统计，但从第 5 个 token 开始，前 4 个样本都是 padding（占位符），只有第 5 个样本有真实 token，统计几乎退化成“单样本 + 一堆 pad 干扰”，可比性大幅下降。

LN 则对每个 token 的特征维单独归一化，
对变长序列更稳健，也更符合 Transformer 的并行计算路径。

---

## 4. Encoder Block
参考： [有深度！Transformer | 万字长文：详细了解前馈神经网络（FFN），内含对大模型的理解](https://zhuanlan.zhihu.com/p/1891081572305846495)
沿用教学示例：

$$
X\in\mathbb{R}^{4\times5}
$$

其中 4 行对应“我/爱/中/国”，5 列是嵌入维度。

### 4.1 一个 Encoder Block 的公式链

1) 多头注意力子层(MHA, multi-head attention)：

$$
Z=\mathrm{MHA}(X),\quad Z\in\mathbb{R}^{4\times5}
$$

2) 残差 + 归一化（LN, Layer Normalization）：

$$
\tilde{X}=\mathrm{LN}(X+Z),\quad \tilde{X}\in\mathbb{R}^{4\times5}
$$

3) 前馈网络（Feed-Forward Network, FFN），也叫多层感知机（MLP）：

$$
F=\mathrm{FFN}(\tilde{X})
$$
先升维，再经过激活函数，再降维；假设中间维度为 2048，则形状：

$$
4\times5\to4\times2048\to4\times5
$$

4) 再做残差 + 归一化：

$$
Y=\mathrm{LN}(\tilde{X}+F),\quad Y\in\mathbb{R}^{4\times5}
$$

结论：一个 Encoder Block 的输入输出 shape 保持一致。

---

## 4.2 FFN 层源码与数学表示

### FFN 源码理解

在 PyTorch 等主流框架中，Transformer 编码器层的 FFN（前馈网络）通常由两层全连接（线性）层组成：

1. 第一层线性变换（升维），常见将特征维度扩展 4 倍；
2. 激活函数（如 ReLU 或 GELU）；
3. 第二层线性变换（降回原始维度）。

### FFN 数学表示

给定输入 \\(X \in \mathbb{R}^{n \times d}\\)（\\(n\\) 为 token 数，\\(d\\) 为特征维度），采用 ReLU 激活函数，FFN 的计算方式如下：

$$
\mathrm{FFN}(X) = \max(0, XW_1 + b_1)W_2 + b_2
$$

其中：

- \\(W_1 \in \mathbb{R}^{d \times 4d}\\)：第一层全连接层权重矩阵，通常将维度扩展 4 倍（经验设置，激活函数会带来信息筛选；若使用 ReLU，负半轴会被截断）；
- \\(b_1 \in \mathbb{R}^{4d}\\)：第一层偏置项；
- \\(W_2 \in \mathbb{R}^{4d \times d}\\)：第二层权重矩阵，将扩展的维度降回 \\(d\\)；
- \\(b_2 \in \mathbb{R}^{d}\\)：第二层偏置项；
- ReLU（或 GELU）：非线性激活函数。

> 目前很多大模型会把偏置项去掉，若无偏置，公式可简化为：
> 
> $$
> \mathrm{FFN}(X) = \mathrm{ReLU}(XW_1)W_2
> $$

FFN 的本质是对每个 token 独立地做两次线性变换和一次激活，增强特征表达能力。

### FFN作用
- 升维层：将输入特征映射到更高维度，使模型能够表达更复杂的特征组合。
- 激活层：引入非线性，使模型能够拟合更复杂的函数关系。
- 降维层：去除冗余信息，浓缩特征，并保持输入输出维度一致。

### 从键值对（KV）理解 FFN
根据前面的介绍，FFN 主要承担“参数化记忆”的作用。这既符合直觉，也和不少研究结论一致。例如：MoE 架构中的专家网络、通用模型中的适配模块（Adapter），以及 LoRA 旁路中的低秩矩阵，本质上都与“线性映射 + 非线性 + 线性映射”的 FFN 思路高度相关。

有一个很有启发的观点是：FFN 的知识可以近似看成 KV Memory 形式的存储。直观上，某些“键向量”对应被激活的模式，而“值向量”承载该模式下的输出偏好。

Attention的基本公式为：
$$
Attention(Q, K, V)=\operatorname{softmax}\left(\frac{QK^\top}{\sqrt{d_k}}\right)V
$$

对比 FFN 的计算公式：
$$
\mathrm{FFN}(X) = \mathrm{ReLU}(XW_1)W_2
$$

如果把 FFN 的 \\(W_1\\) 看作“键矩阵”，\\(W_2\\) 看作“值矩阵”，并把 softmax 与激活函数都抽象为某种选择函数 \\(f\\)，可写成近似形式：
$$
\mathrm{FFN}(X) = f(xK^T)V
$$

输入 \\(x\\) 与每个 \\(k_i\\) 点乘后得到系数（memory coefficient），再对对应的 \\(v_i\\) 加权求和得到输出。这个过程与 QKV 点乘注意力有相似性，但两者有关键区别：

- Attention 的计算是 context-dependent 的：\\(Q,K,V\\) 都来自当前输入表示，随上下文变化；FFN 的“KV”是 context-independent 的，来自固定的可学习参数矩阵。

- Attention 常用 softmax（normalized），而 FFN 常用 ReLU/GELU（unnormalized）。

总结：Attention 更偏向“按上下文检索即时信息”，FFN 更偏向“在参数中沉淀通用模式”。这也解释了为什么即便上下文窗口有限，模型仍能保留训练语料中的大量统计知识。

### FFN 与 Attention 的非线性分工
Attention 通过 softmax 引入非线性变换，那么 FFN 为什么还要引入非线性呢？

根据公式，Attention 中 softmax 对 \\(q\\) 和 \\(k\\) 的匹配分数进行非线性变换；但对 \\(v\\) 本身并不做逐元素非线性处理，每次计算更接近“对 value 向量做加权平均”。

另外，softmax 的核心作用是权重归一化，而 FFN 中 ReLU/GELU 的核心作用是提升特征变换的表达能力。两类非线性作用位置不同、目的也不同。

## 5. 从 Block 到 Encoder

若堆叠 6 个 Encoder Block：

$$
X^{(0)}=X,
\quad
X^{(l)}=\mathrm{EncoderBlock}^{(l)}\big(X^{(l-1)}\big),\ l=1,\dots,6
$$

最终输出：

$$
\mathrm{Memory}=X^{(6)}\in\mathbb{R}^{4\times5}
$$

这个 Memory（编码信息矩阵）会传给 Decoder。

---

## 6. 小结

1) BN 与 LN 的关键差异在于统计维度：BN 依赖跨样本统计，LN 通常沿特征维做逐 token 归一化。  
2) NLP 场景中，变长序列与 padding 使 BN 的统计更容易受干扰，因此 LN 更常见。  
3) 一个 Encoder Block 内部经历 MHA、残差加 LN、FFN、残差加 LN，输入输出 shape 保持一致。  
4) 多层堆叠后得到的 Memory 是 Decoder 的核心条件输入。
