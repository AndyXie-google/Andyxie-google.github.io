---
title: dev-sidecar补充_安装、证书与反向代理排障
date: 2025-04-23
tags: [dev-sidecar, 代理, GitHub, Jetson]
series: PaddlePaddle工程实战连载
status: 可发布
---

## 目录
- [目录](#目录)
- [`dev-sidecar`及其安装](#dev-sidecar及其安装)
- [快速使用步骤](#快速使用步骤)
- [相关链接](#相关链接)

---
  - 注意：dev-sidecar 需要必须安装证书才能代理网络流量

## `dev-sidecar`及其安装
[`dev-sidecar`](https://github.com/docmirror/dev-sidecar)(开发者边车)利用本地反向代理, 解决github打不开, 进行github访问加速, git clone加速, git release下载加速, stackoverflow加速; 

> [!TIP]
> 本文适合需要稳定访问 GitHub 与加速下载的场景。

> [!IMPORTANT]
> `dev-sidecar`是否可用，核心在证书是否正确导入到系统与浏览器。

## 快速使用步骤
1. 安装 dev-sidecar（arm64 对应 deb 包）。
2. 启动后安装/导入证书（系统与浏览器都要检查）。
3. 代理服务与系统代理会自动开启
4. 电脑关机前需要先手动关闭`dev-sidecar`, 否则下次开机后代理异常, 无法正常上网
5. 建议开启代理`git.exe`选项, 提升`git clone`速度(当然也可以访问镜像站)

> <mark>如果只安装程序不安装证书，通常会出现“代理已开但加速无效”的情况。</mark>

> ⚠️ 若出现网络异常，先临时关闭系统代理验证是否为代理链路问题。

---

## 相关链接
- dev-sidecar 下载: (https://sourceforge.net/projects/dev-sidecar.mirror/)
- dev-sidecar 项目说明: (https://github.com/docmirror/dev-sidecar)
