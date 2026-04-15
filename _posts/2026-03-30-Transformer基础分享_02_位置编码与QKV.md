---
title: Transformer基础分享_02_位置编码与QKV
date: 2026-03-30
tags:
  - Transformer
  - PositionalEncoding
  - SelfAttention
  - 深度学习入门
series: Transformer基础分享
status: 可发布
---

<!-- # Transformer 基础分享：位置编码与自注意力 -->
另见[LLM学习记录（五）--超简单的RoPE理解方式](https://zhuanlan.zhihu.com/p/642289220)

> 目标：回答三个核心问题。  
> 1) 词嵌入后为什么还不够？  
> 2) 位置编码为什么要用 sin/cos？  
> 3) 自注意力里 Q/K/V 是怎么来的？

---

## 1. 核心结论先看

| 问题 | 结论 | 一句话理解 |
|---|---|---|
| 词嵌入后能直接进 Transformer 吗？ | 不能直接只用词嵌入 | 还需要显式注入位置信息 |
| 位置编码做什么？ | 给每个 token 注入顺序线索 | 让模型区分“我爱中国”与“中国爱我” |
| 输入到底是什么？ | \\(X=E+P\\) | 词嵌入矩阵与位置编码矩阵逐元素相加 |
| Q/K/V 怎么来？ | \\(Q=XW_Q,\ K=XW_K,\ V=XW_V\\) | 由输入线性映射得到三组表征 |

---

## 2. 为什么词嵌入后还不够

前面我们已经有词嵌入了，但这一步只解决了“**可计算**”，还没有完全解决“**顺序信息**”。

在不显式注入位置信息时，Transformer 很难仅靠输入本身区分词序差异。  
因此，**位置信息必须作为额外信号注入**。

> 直观说法：
> “我爱中国”和“中国爱我”字集合相同，但语义不同，差异来自顺序。

---

## 3. 一个直观想法：直接加位置编号

一个常见直觉是：

- 原词嵌入是 \\(4\times5\\)；
- 再额外拼一列位置编号（1,2,3,4）；
- 变成 \\(4\times6\\) 送入模型。

这个想法并非完全不可用，但有明显局限：

- 位置值线性增长，尺度随长度变化明显；
- 对长句外推时，可能出现分布偏移；
- 位置之间的相对关系不够平滑。

所以实践中更常见的是 **Positional Encoding（PE）**。

> 口语化例子补充：
> 你可以把它理解成“给每个字贴序号标签”。比如“我爱中国”里，
> “我/爱/中/国”对应位置是 1/2/3/4（或 0/1/2/3）。
> 这个想法直觉上没错，但直接用线性编号会让数值随句长单调变大，
> 长句场景下常常不够稳健。

---

## 4. Positional Encoding 的数学形式（LaTeX）

设模型维度为 \\(d_{model}\\)，位置为 \\(pos\\)，维度索引为 \\(i\\)，经典正余弦编码为：

$$
PE(pos,2i)=\sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

$$
PE(pos,2i+1)=\cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)
$$

这组设计的优点：

- 不同频率覆盖不同尺度的位置变化；
- 相对位移可以通过线性关系表达；
- 长度外推通常更稳定。

具体计算: 
- 设句子“我爱中国”，位置从 0 开始编号，即 \\(pos\in\{0,1,2,3\}\\)；
- 假设嵌入维度 \\(d_{model}=5\\)（教学简化）；
- 当看第一个字“我”时，\\(pos=0\\)，于是很多分量会落在 \\(\sin(0)=0,\cos(0)=1\\) 这类基准值附近；
- 当看第二个字“爱”时，\\(pos=1\\)，对应会出现类似 \\(\sin(1)\approx0.84\\) 的数值变化。

这就是“同一维度上，不同位置会有不同编码”的直观来源。

> 也可以使用可学习位置编码（learned positional embedding），二者在不同任务中各有取舍。

---

## 5. 结合例子：我爱中国（\\(4\times5\\)）

句子按汉字粒度切分：

$$
[\text{我},\ \text{爱},\ \text{中},\ \text{国}]
$$

若词嵌入矩阵记为 \\(E\in\mathbb{R}^{4\times5}\\)，位置编码矩阵记为 \\(P\in\mathbb{R}^{4\times5}\\)，则输入为：

$$
X=E+P,\quad X\in\mathbb{R}^{4\times5}
$$

也就是：

- 行数 4 对应 4 个 token；
- 列数 5 对应每个 token 的嵌入维度。

---

## 6. SVG 示意图 1：输入构造（硬编码）

<svg width="100%" viewBox="0 0 660 180" xmlns="http://www.w3.org/2000/svg" aria-label="X equals E plus P" role="img" preserveAspectRatio="xMidYMid meet">
 <title>X = E + P</title>
 <desc>词嵌入矩阵 E 与位置编码矩阵 P 相加，得到模型输入 X，三者形状均为 4×5。</desc>
 <defs>
  <marker orient="auto" refY="3" refX="8" markerHeight="10" markerWidth="10" id="arr">
   <path id="svg_1" fill="#1f2937" d="m0,0l8,3l-8,3l0,-6z"/>
  </marker>
 </defs>
 <g>
  <title>Layer 1</title>
  <text id="svg_3" font-weight="700" fill="#111827" font-family="Georgia, serif" font-size="28" y="33" x="31">词嵌入 E</text>
  <rect stroke="#2f4f6f" id="svg_4" stroke-width="2" fill="#e5eef8" rx="10" height="113" width="152" y="52" x="9"/>
  <text id="svg_5" fill="#1f2937" font-family="Times New Roman, serif" font-size="30" y="117" x="55">4 × 5</text>
  <text id="svg_6" fill="#374151" font-family="Times New Roman, serif" font-size="64" y="128" x="181">+</text>
  <text id="svg_7" font-weight="700" fill="#111827" font-family="Georgia, serif" font-size="28" y="33" x="248">位置编码 P</text>
  <rect stroke="#815b2f" id="svg_8" stroke-width="2" fill="#f8ead9" rx="10" height="114" width="150" y="53" x="236"/>
  <text id="svg_9" fill="#1f2937" font-family="Times New Roman, serif" font-size="30" y="118" x="280">4 × 5</text>
  <line id="svg_10" marker-end="url(#arr)" stroke-width="3" stroke="#1f2937" y2="106" x2="501" y1="106" x1="396"/>
  <text id="svg_11" font-weight="700" fill="#111827" font-family="Georgia, serif" font-size="28" y="33" x="506">模型输入 X</text>
  <rect stroke="#2f6f4f" id="svg_12" stroke-width="2" fill="#dff3e7" rx="10" height="117" width="133" y="49" x="510"/>
  <text id="svg_13" fill="#1f2937" font-family="Times New Roman, serif" font-size="30" y="117" x="547">4 × 5</text>
 </g>
</svg>

---

## 7. 自注意力开场：Q/K/V 从哪里来

| 符号 | 常见称呼 | 直觉作用 |
|---|---|---|
| Q | Query | 我想查询什么信息 |
| K | Key | 我可以提供什么索引 |
| V | Value | 我真正携带的内容 |

得到输入矩阵 \\(X\\) 后，自注意力第一步是线性映射：

$$
Q=XW_Q,\qquad K=XW_K,\qquad V=XW_V
$$

若本讲示例取：

- \\(X\in\mathbb{R}^{4\times5}\\)
- \\(W_Q,W_K,W_V\in\mathbb{R}^{5\times5}\\)

则有：

$$
Q,K,V\in\mathbb{R}^{4\times5}
$$

按行看，每个 token 都会得到一组 \\((q_i,k_i,v_i)\\)，其中 \\(q_i,k_i,v_i\in\mathbb{R}^{1\times5}\\)。

继续沿用“我爱中国”的口语化拆解：

- 第一行（“我”）记为 \\(x_1\in\mathbb{R}^{1\times5}\\)；
- 通过三组参数分别映射：\\(q_1=x_1W_Q,\ k_1=x_1W_K,\ v_1=x_1W_V\\)；
- 同理可得“爱/中/国”对应的 \\((q_2,k_2,v_2),(q_3,k_3,v_3),(q_4,k_4,v_4)\\)。

“三分天下”其实就是这一步：同一个输入行向量，被投影成三种不同“视角”的表示。



<svg viewBox="0 0 980 340" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="X to separate Q K V matrices" preserveAspectRatio="xMidYMid meet">
 <title>X 到分离的 Q/K/V 矩阵映射</title>
 <desc>左侧输入 X 的四行 token 分别映射到右侧分开的 Q、K、V 三个矩阵。</desc>
 <defs>
  <marker orient="auto" refY="3" refX="8" markerHeight="10" markerWidth="10" id="arrMerge">
   <path id="svg_1" fill="#111827" d="m0,0l8,3l-8,3l0,-6z"/>
  </marker>
 </defs>
 <g>
  <title>Layer 1</title>
  <rect stroke="#355070" id="svg_3" stroke-width="2" fill="#e8f1ff" rx="10" height="140" width="163" y="115.00001" x="38"/>
  <text id="svg_5" font-size="18" font-family="Consolas, monospace" fill="#0f172a" y="145" x="85">x1  (我)</text>
  <text id="svg_6" font-size="18" font-family="Consolas, monospace" fill="#0f172a" y="178" x="85">x2  (爱)</text>
  <text id="svg_7" font-size="18" font-family="Consolas, monospace" fill="#0f172a" y="211" x="85">x3  (中)</text>
  <text id="svg_8" font-size="18" font-family="Consolas, monospace" fill="#0f172a" y="244" x="85">x4  (国)</text>
  <line id="svg_9" marker-end="url(#arrMerge)" stroke-width="3" stroke="#111827" y2="90" x2="330" y1="140" x1="220"/>
  <line id="svg_10" marker-end="url(#arrMerge)" stroke-width="3" stroke="#111827" y2="180" x2="330" y1="180" x1="220"/>
  <line id="svg_11" marker-end="url(#arrMerge)" stroke-width="3" stroke="#111827" y2="270" x2="330" y1="220" x1="220"/>
  <text id="svg_12" font-size="16" font-family="Consolas, monospace" fill="#111827" y="84" x="240">× W_Q</text>
  <text id="svg_13" font-size="16" font-family="Consolas, monospace" fill="#111827" y="172" x="240">× W_K</text>
  <text id="svg_14" font-size="16" font-family="Consolas, monospace" fill="#111827" y="286" x="240">× W_V</text>
  <rect id="svg_15" stroke-width="2" stroke="#2d6a4f" fill="#eafaf0" rx="10" height="140" width="180" y="20" x="350"/>
  <text id="svg_16" font-size="30" font-family="Times New Roman, serif" fill="#0f172a" text-anchor="middle" y="52" x="440">Q (4×5)</text>
  <text id="svg_17" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="84" x="372">q1 = x1WQ</text>
  <text id="svg_18" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="108" x="372">q2 = x2WQ</text>
  <text id="svg_19" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="132" x="372">q3 = x3WQ</text>
  <text id="svg_20" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="156" x="372">q4 = x4WQ</text>
  <rect id="svg_21" stroke-width="2" stroke="#9c6644" fill="#fff4e6" rx="10" height="140" width="180" y="100" x="350"/>
  <text id="svg_22" font-size="30" font-family="Times New Roman, serif" fill="#0f172a" text-anchor="middle" y="132" x="440">K (4×5)</text>
  <text id="svg_23" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="164" x="372">k1 = x1WK</text>
  <text id="svg_24" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="188" x="372">k2 = x2WK</text>
  <text id="svg_25" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="212" x="372">k3 = x3WK</text>
  <text id="svg_26" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="236" x="372">k4 = x4WK</text>
  <rect id="svg_27" stroke-width="2" stroke="#6d597a" fill="#f3e8ff" rx="10" height="140" width="180" y="180" x="350"/>
  <text id="svg_28" font-size="30" font-family="Times New Roman, serif" fill="#0f172a" text-anchor="middle" y="212" x="440">V (4×5)</text>
  <text id="svg_29" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="244" x="372">v1 = x1WV</text>
  <text id="svg_30" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="268" x="372">v2 = x2WV</text>
  <text id="svg_31" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="292" x="372">v3 = x3WV</text>
  <text id="svg_32" font-size="16" font-family="Consolas, monospace" fill="#0f172a" y="316" x="372">v4 = x4WV</text>
  <line id="svg_33" marker-end="url(#arrMerge)" stroke-width="3" stroke="#111827" y2="170" x2="660" y1="170" x1="540"/>
  <text id="svg_34" font-size="16" font-family="Consolas, monospace" fill="#111827" y="158" x="528">组合进入注意力计算</text>
  <rect id="svg_35" stroke-width="2" stroke="#b45309" fill="#fef3c7" rx="10" height="100" width="280" y="120" x="670"/>
  <text id="svg_36" font-size="18" font-family="Consolas, monospace" fill="#111827" text-anchor="middle" y="158" x="810">Attention(Q,K,V)</text>
  <text id="svg_37" font-size="16" font-family="Consolas, monospace" fill="#334155" text-anchor="middle" y="186" x="810">softmax(QK^T / sqrt(d_k))V</text>
  <text style="cursor: move;" id="svg_38" font-size="30" font-family="Times New Roman, serif" fill="#0f172a" text-anchor="middle" y="108" x="120">X (4×5)</text>
 </g>
</svg>

---

## 8. 过渡到下一节

到这里，我们已经完成了从“词向量”到“可进入注意力计算的三组表示”的准备工作。  
下一步自然是：

1. 计算注意力分数矩阵 \\(QK^\top\\)；
2. 做缩放与 softmax；
3. 用权重对 \\(V\\) 加权求和，得到新的上下文表示。

---

## 10. 一句话收束

**位置编码解决“顺序感知”，Q/K/V 解决“信息交互”，两者叠加才构成 Transformer 能理解句子的关键入口。**

---

## 系列导航

- 上一篇：[Transformer基础分享_01_词嵌入](./2026-03-30-Transformer基础分享_01_词嵌入.md)
- 下一篇：[Transformer基础分享_03_自注意力计算与直觉](./2026-03-30-Transformer基础分享_03_自注意力计算与直觉.md)
