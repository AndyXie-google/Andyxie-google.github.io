---
title: Transformer基础分享_08_BERT与ViT
date: 2026-04-11
tags:
  - Transformer
  - BERT
  - ViT
  - 预训练模型
  - 视觉Transformer
status: 可发布
---

**另见: [详解 GPT 生成机制中的 KV Cache](https://zhuanlan.zhihu.com/p/687020859)**

本文聚焦两个经典方向: NLP 中的 BERT 与 CV 中的 ViT。目标是从 Transformer 统一视角出发, 说明两者的输入形式、训练任务与分类流程, 并解释它们与 CNN 的差异。

## 1. BERT 简介与原理

BERT (Bidirectional Encoder Representations from Transformers) 是由多层 Transformer 编码器堆叠形成的预训练模型。Transformer 原始结构由编码器和解码器组成, BERT 仅保留编码器部分。

可以粗略理解为:

- 仅保留编码器, 发展为 BERT 这类双向表征模型。
- 仅保留解码器, 发展为 GPT (Generative Pretrained Transformer) 这类自回归生成模型。

BERT 预训练阶段的核心任务主要包括以下两类。

### 1.1 Masked Language Model（MLM，完形填空）

- 随机遮盖输入序列中的部分词元, 让模型预测被遮住的内容。
- 例如: 输入“我爱中国”, 将“国”替换为 `[MASK]`, 得到“我爱中[MASK]”, 再让模型恢复该位置。
- 若输入张量形状为\\(4\times5\\), 编码器输出仍为\\(4\times5\\); 经过词表投影后可得到\\(4\times5000\\) (假设词表大小为 5000)。
- 对每个位置做 softmax 预测, 训练时主要在被 mask 的位置计算损失。
- 虽然只在部分位置计算损失, 但每个位置的表示都通过自注意力融合了全局上下文信息。

### 1.2 Next Sentence Prediction（NSP，句子关系判断）

- 目标是判断两个句子在原始语料中是否相邻。
- 输入形式通常为: `[CLS] 句子A [SEP] 句子B [SEP]`。
- 编码器输出后, 取 `[CLS]` 位置向量送入二分类头, 输出“相邻/不相邻”。
- 标签一般用 1 表示相邻, 0 表示不相邻。
- 训练中通常将 MLM 与 NSP 联合优化, 其中 MLM 的遮盖比例常设为 15%。
- 该预训练流程不依赖人工逐条标注, 主要依赖大规模高质量文本语料 (如新闻、百科等)。

## 2. ViT（Vision Transformer）简介与原理

ViT 是将 Transformer 引入计算机视觉 (CV) 的代表性模型之一。核心思路是把图像切成 patch 序列, 再按“词序列”方式送入编码器。

### 2.1 ViT 的输入处理

- 将图像划分为若干 patch。示例中, 一张\\(28\times28\\) 图像被切为 4 个\\(14\times14\\) patch。
- 每个 patch 先拉平为一维向量 (如\\(14\times14=196\\)), 再通过线性层映射到嵌入维度\\(d\\) (如示意用\\(d=5\\))。
- 所有 patch 的嵌入按序拼接, 可形成类似\\(4\times5\\) 的输入矩阵 (示意维度)。

#### ViT Patch 划分与嵌入流程示意

<svg width="650" height="240" viewBox="0 0 650 240" xmlns="http://www.w3.org/2000/svg" style="display:block;margin:16px auto;background:#f8fafc">
  <!-- 原图像 -->
  <rect x="30" y="40" width="100" height="100" fill="#e0e7ff" stroke="#6366f1" stroke-width="2"/>
  <text x="80" y="35" text-anchor="middle" font-size="14" fill="#334155">原图片</text>
  <!-- Patch 划分 -->
  <rect x="30" y="40" width="50" height="50" fill="none" stroke="#6366f1" stroke-width="2" stroke-dasharray="4 2"/>
  <rect x="80" y="40" width="50" height="50" fill="none" stroke="#6366f1" stroke-width="2" stroke-dasharray="4 2"/>
  <rect x="30" y="90" width="50" height="50" fill="none" stroke="#6366f1" stroke-width="2" stroke-dasharray="4 2"/>
  <rect x="80" y="90" width="50" height="50" fill="none" stroke="#6366f1" stroke-width="2" stroke-dasharray="4 2"/>
  <text x="80" y="160" text-anchor="middle" font-size="12" fill="#64748b">分为4个patch</text>
  <!-- Patch 拉平箭头 -->
  <line x1="55" y1="140" x2="55" y2="180" stroke="#64748b" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="105" y1="140" x2="105" y2="180" stroke="#64748b" stroke-width="2" marker-end="url(#arr)"/>
  <defs>
    <marker id="arr" markerWidth="10" markerHeight="10" refX="8" refY="3" orient="auto">
      <path d="M0,0 L8,3 L0,6 Z" fill="#64748b"/>
    </marker>
  </defs>
  <!-- Patch 拉平成向量 -->
  <rect x="40" y="180" width="30" height="20" fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>
  <rect x="90" y="180" width="30" height="20" fill="#f1f5f9" stroke="#64748b" stroke-width="1.5"/>
  <text x="55" y="195" text-anchor="middle" font-size="11" fill="#334155">拉平</text>
  <text x="55" y="208" text-anchor="middle" font-size="10" fill="#64748b">196维</text>
  <text x="105" y="195" text-anchor="middle" font-size="11" fill="#334155">拉平</text>
  <text x="105" y="208" text-anchor="middle" font-size="10" fill="#64748b">196维</text>
  <!-- 嵌入映射箭头 -->
  <line x1="70" y1="190" x2="150" y2="190" stroke="#64748b" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="120" y1="190" x2="200" y2="190" stroke="#64748b" stroke-width="2" marker-end="url(#arr)"/>
  <!-- 嵌入向量 -->
  <rect x="150" y="180" width="30" height="20" fill="#fef9c3" stroke="#eab308" stroke-width="1.5"/>
  <rect x="200" y="180" width="30" height="20" fill="#fef9c3" stroke="#eab308" stroke-width="1.5"/>
  <text x="165" y="195" text-anchor="middle" font-size="11" fill="#b45309">嵌入</text>
  <text x="165" y="208" text-anchor="middle" font-size="10" fill="#eab308">5维</text>
  <text x="215" y="195" text-anchor="middle" font-size="11" fill="#b45309">嵌入</text>
  <text x="215" y="208" text-anchor="middle" font-size="10" fill="#eab308">5维</text>
  <!-- 省略号 -->
  <text x="250" y="190" text-anchor="middle" font-size="18" fill="#64748b">...</text>
  <!-- Patch 嵌入合并为矩阵 -->
  <rect x="300" y="120" width="120" height="60" fill="#e0f2fe" stroke="#0ea5e9" stroke-width="2"/>
  <text x="360" y="146" text-anchor="middle" font-size="13" fill="#0369a1">patch嵌入矩阵</text>
  <text x="360" y="164" text-anchor="middle" font-size="13" fill="#0369a1">4×5</text>
  <line x1="180" y1="190" x2="300" y2="150" stroke="#0ea5e9" stroke-width="2" marker-end="url(#arr)"/>
  <line x1="230" y1="190" x2="300" y2="170" stroke="#0ea5e9" stroke-width="2" marker-end="url(#arr)"/>
</svg>

> **补充说明：**
> - 每个 patch 先拉平为\\(h\times w\\) 维 (如\\(14\times14=196\\)), 再经线性映射得到\\(d\\) 维嵌入。
> - 文中\\(d=5\\) 仅用于示意, 实际常见维度会更大。
> - 拉平是重排, 嵌入是特征映射, 两者作用不同。

### 2.2 ViT 的分类任务流程

- 将 patch 嵌入序列输入 Transformer 编码器。
- 在序列最前面拼接一个可学习的分类标记 (class token), 例如从\\(4\times5\\) 扩展为\\(5\times5\\) (示意)。
- 编码后取第一个位置 (class token) 的输出向量, 接分类头进行二分类或多分类。
- 经过多层自注意力后, 该向量能够聚合全局 patch 信息。

### 2.3 ViT 与 CNN 的对比

- 在大规模数据集 (如 ImageNet) 上, ViT 常能达到或超过 CNN (如 ResNet) 的效果。
- 在中小规模数据场景中, CNN 的局部感受野与平移不变性等归纳偏置通常更有优势。
- Transformer 结构更通用、先验偏置更弱, 往往需要更多数据或更强预训练支持。

## 3. 总结

- BERT 采用编码器堆叠, GPT 采用解码器堆叠。
- BERT 的代表性预训练任务是 MLM 与 NSP。
- ViT 将图像 patch 序列化, 并借助 class token 完成分类。
- 模型选择与数据规模强相关: 大数据场景下 Transformer 优势更明显, 小数据场景中 CNN 仍具竞争力。
