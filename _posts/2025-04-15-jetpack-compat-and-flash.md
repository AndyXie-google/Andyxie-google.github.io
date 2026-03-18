---
title: jetpack相关问题补充_版本兼容与刷机选择
date: 2025-04-15
tags: [Jetson, JetPack, 刷机, 兼容性]
series: PaddlePaddle工程实战连载
status: 可发布
---

> ⚠️ 刷机涉及固件与镜像匹配，操作前请先确认板卡版本信息。

## 目录
- [目录](#目录)
    - [什么是`jetpack`](#什么是jetpack)
    - [刷机注意事项: 更新固件](#刷机注意事项-更新固件)
    - [相关链接](#相关链接)

---

<a id="what-is-jetpack"></a>
#### 什么是`jetpack`
`jetpack`是`nvdia`为`jetson`系列开发板制作的官方`ubuntu_arm`镜像, 内置`cuda`, `cudnn`, `opencv`, `tensorrt`等gpu调用相关组件;
如jetpack 6.2版本配置如下: `CUDA 12.6, TensorRT 10.3, cuDNN 9.3, VPI 3.2, DLA 3.1, DLFW 24.0`

> [!NOTE]
> <mark>`jetpack`本质上是系统镜像 + AI 基础组件打包，不只是单一软件安装包。</mark>

<a id="flash-notes"></a>
#### 刷机注意事项: 更新固件
先看固件版本, 在2025年之前出厂的jetson orin nano可能带有较老的`L4T`出厂固件, 如果固件版本小于`36.x`, 如果不更新固件只能烧录`jetpack 5.x`及以下的jetpack镜像;
更新固件请看[Initial Setup Guide for Jetson Orin Nano Developer Kit](https://www.jetson-ai-lab.com/tutorials/initial-setup-jetson-orin-nano/), 其实就是先准备一张>64GB的microSD卡, 先用`sdcard formatter`格式化, 再使用刷写软件(如balena_etcher)烧入[jetpack 5.1.3](https://developer.nvidia.com/downloads/embedded/l4t/r35_release_v5.0/jp513-orin-nano-sd-card-image.zip)镜像, 软重启后即可烧入`jetpack 6.x`版本
推荐烧入jetpack 6.2及以上固件, 可以启用25W以上功耗模式, 性能更强

> [!IMPORTANT]
> <mark>先看固件版本再选镜像版本，是避免启动循环问题的关键步骤。</mark>

---

<a id="links"></a>
#### 相关链接
- jetpack介绍: https://docs.nvidia.cn/jetson/jetpack/introduction/index.html
- Jetson Orin Nano Getting Started: https://developer.nvidia.com/embedded/learn/get-started-jetson-orin-nano-devkit#firmware
- JetPack 6.2 镜像: https://developer.nvidia.com/downloads/embedded/l4t/r36_release_v4.3/jp62-orin-nano-sd-card-image.zip