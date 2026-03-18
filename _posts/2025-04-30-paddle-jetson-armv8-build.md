---
title: paddle+jetson端侧目标识别模型部署(1)_paddlepaddle-gpu在armv8架构开发板上的编译
date: 2025-04-30
tags: [PaddlePaddle, Jetson, 工程化, 部署]
series: PaddlePaddle工程实战连载
status: 可发布
---

### `paddle+jetsonorin`端侧目标识别模型部署(1): `paddlepaddle-gpu`在`armv8`架构开发板上的编译

> [!TIP]
> 本文为实操记录向内容，包含环境、编译命令与报错处理。

> [!WARNING]
> 所有命令基于文中环境版本，不同版本组合请先做兼容性确认。

---

#### 为什么要编译
如果想要在`x86_64`的PC端或没有独立`gpu`的`arm`开发板部署paddlepaddle框架, 可以直接前往[飞桨快速安装](https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html), 无需编译

如果nvidia嵌入式开发板上面安装了`jetpack 5-1-2`(`jetpack`相关问题, 前往[这里](./01A_jetpack版本兼容与刷机说明.md))可以安装官方已经编译好的[`paddlepaddle-gpu 3.0.0`whl文件](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/download_lib.html#python)

如果不属于以上两种情况, 又想使用gpu加速推理, 就只能自行通过源码编译出`whl`文件了

> [!NOTE]
> <mark>如果不属于以上两种情况, 又想使用gpu加速推理, 就只能自行通过源码编译出`whl`文件了</mark>

---

#### 编译环境
**硬件配置**: `jetson orin nano 4g/8g(super)` 开发板一块
**软件配置**(可以跟我的不一样): 
  - `jetpack 6.2`(内置`cuda 12.6`, `cudnn 9.3`, `ubuntu 22.04`)
  - `python 3.10`(ubuntu 22.04系统自带)
  - `python 3.11.5(anaconda)`(与系统python环境隔离, 配置独立编译环境)
  - `cmake 3.19`, 安装方式: `sudo apt install cmake`或者从[官网](https://cmake.org/download/)下载[`binary_package`](https://ghfast.top/https://github.com/Kitware/CMake/releases/download/v3.19.8/cmake-3.19.8-Linux-aarch64.sh)

**编译加速**(可选):
  - 多核并行编译(必选, 但是内核数指定过多容易爆内存, 需要`swap`"增加"内存)： `make -j N`
  - 编译缓存工具： `ccache`
  - 分布式编译工具：`distcc`

---


#### `swap`将ssd硬盘空间划分为虚拟内存
```bash
df -h # 查看当前机器的剩余磁盘空间
```

`terminal`返回数据: 
```bash
/dev/nvme0n1 256G 0 256G 0% /mnt   
```

```bash
dd if=/dev/nvme0n1 of=/swapfile bs=1M count=32768 # 划分32G作为虚拟内存
mkswap /swapfile
swapon /swapfile
free -h # 设置完成, 查看当前swap使用情况
```

> [!WARNING]
> 涉及磁盘与内存操作，执行前请确认设备与路径。

---

#### 开始编译
<mark>如果编译出错建议完整清除build文件夹, 否则新一次编译可能再次报错!</mark>
```bash
rm -rf ./build
mkdir build
cd build
```

完整编译过程:
```bash
cd /home/smartcar/Downloads/Paddle
# 如果想要从头开始, 删除旧的build文件夹
# rm -rf ./build
mkdir build
cd build

# 10分钟, 这里可能老是报连接超时, 尝试连接手机用移动数据开热点, 下面指令多跑几遍
"/home/smartcar/Downloads/cmake-3.19.8-Linux-aarch64/bin/cmake" .. -DWITH_CONTRIB=OFF -DWITH_MKL=OFF -DWITH_MKLDNN=OFF -DWITH_TESTING=OFF -DCMAKE_BUILD_TYPE=Release -DON_INFER=ON -DWITH_PYTHON=ON -DWITH_XBYAK=OFF -DWITH_NV_JETSON=ON -DWITH_ARM=ON -DWITH_NCCL=OFF -DPY_VERSION=3.10 -DCUDA_ARCH_NAME=Auto -DCMAKE_CXX_FLAGS="-Wno-error=class-memaccess" -DCMAKE_CXX_FLAGS='-Wno-error -w'

# 编译时间10小时
ulimit -n 102400  # 提高文件打开个数上限, 防止报错: "Too many open files"
make -j6  # jetson orin nano共6核, 且不能超线程; 
# 如果想要关闭ssh也能继续编译用 nohup make -j6
```

---

#### 可能出现的问题
**错误1**: github连接不畅, 编译的时候老是报`connection timeout`, 没啥别的办法, 下载是概率性问题, 多跑几次cmake命令就下全了, 或者戳这里看如何[安装dev-sidecar并设置反向代理](./01B_dev-sidecar安装与反向代理说明.md)

**错误2**: 老是弹`c++: error: unrecognized command-line option ‘-mmmx’`之类的错误, 但是实际上gcc编译器支持`mmmx`指令集; 原因: 没有禁用 NCLL : `-DWITH_NCCL=OFF`

**错误3**: make完成后报
```
Traceback (most recent call last): File "/home/smartcar/Downloads/Paddle/build/python/setup.py", line 1008, in <module> shutil.copy('/home/smartcar/Downloads/Paddle/build/third_party/install/flashattn/lib/libflashattn.so', libs_path) File "/usr/lib/anaconda3/lib/python3.11/shutil.py", line 419, in copy copyfile(src, dst, follow_symlinks=follow_symlinks) File "/usr/lib/anaconda3/lib/python3.11/shutil.py", line 256, in copyfile with open(src, 'rb') as fsrc: ^^^^^^^^^^^^^^^ FileNotFoundError: [Errno 2] No such file or directory: '/home/smartcar/Downloads/Paddle/build/third_party/install/flashattn/lib/libflashattn.so' m
```
推测可能是前几次试编译命令的时候混入了`-DWITH_NCCL=ON`的内容, 导致有一部分内容不需要编译(后面用的是`-DWITH_NCCL=OFF`), 但是在最后合成`.whl`包的时候找不到软链接`.so`文件

因此只能从头来过, 这一部分是在板卡外接显示屏/键盘上完成的, 没有用`ssh`, 从`25/04/28`开始挂了一个晚上

**错误4**: 今天(25/04/29)挂了一个上午, 上完课回来发现报了一个错: `httpx not found`并且建议我使用`pip install -r python/requirements.txt `把第三方库都装好(先`cd ../`从`./Paddle/build`目录退回`./Paddle`目录), 试了一下发现`requirements satisfied`, 没有缺少组件, 然后发现Terminal启动了conda环境, 也就是说之前在`3.11.5`的`base`环境下进行编译, 因此先退出当前conda环境, 在系统python环境下补全第三方库再使用系统环境重新编译;
```bash
# 完整指令
conda deactivate

cd ../
pip install -r python/requirements.txt
cd ./build

make -j6
```
---

#### 参考资料
- 感谢飞桨官方的[linux编译指南](https://www.paddlepaddle.org.cn/inference/v3.0/guides/install/compile/source_compile_under_Linux.html)
- 感谢知乎大佬`pprffh`[Jetson Jetpack6.2中编译安装基于Python API的 Paddle Inference GPU 3.0 (同时生成C/C++ API安装包)](https://zhuanlan.zhihu.com/p/30190018963)的编译教程
- 感谢[JetsonOrin源码安装部署PaddlePaddle](https://blog.csdn.net/qq_38418182/article/details/146257152)
- 感谢[[C++] 提高 C/C++ 项目编译速度的神兵利器](https://zhuanlan.zhihu.com/p/11165843086)
- 感谢[Ubuntu20.04配置distcc(联合编译/分布式编译)](https://blog.csdn.net/weixin_42219627/article/details/120102685)
- 感谢[【linux添加虚拟内存（swap）】](https://blog.csdn.net/dmonsterer/article/details/138960821)

<!-- ## 四、从“能编译”到“能接项目”还差一步

【来源位置：学习日记.md，日期：25/6/22】
> /paddle_jetson/base/infer_wrap.py 物体识别

【来源位置：学习日记.md，日期：25/6/22】
> 强制设定路径：
> model_path = 'paddle_jetson/lane_model/cnn_lane.pdmodel'
> params_path = 'paddle_jetson/lane_model/cnn_lane.pdiparams'

这一步非常真实：当模型真正接入项目，最先炸的往往不是算法，而是路径、资源、运行时显示环境这类“系统侧细节”。

你后续记录里还出现了显示变量问题：

【来源位置：学习日记.md，日期：25/6/22】
> qt.qpa.xcb could not connect to display
> export DISPLAY=localhost:10.0

这说明部署不是单点任务，而是软硬件共同体。工程一致性的含义也在这里：不仅模型版本一致，运行条件也要一致。

## 五、本篇给一个可以落地的最小清单

下面这份清单可以直接作为你后续每篇的开头模板：

1. 固定版本矩阵：硬件、系统、驱动、Python、Paddle。
2. 固定构建参数：尤其是 Jetson 上的编译开关。
3. 固定依赖清单：requirements、pip 源、离线包策略。
4. 固定资源路径：模型文件、配置文件、日志路径。
5. 固定排障顺序：先环境，再依赖，再参数，再业务。

只要这五件事固定住，后面的“训练替换模型”“升级框架版本”“换新板卡”都会轻松很多。

## 六、结语

PaddlePaddle 本身不难，难的是你把它放进真实项目后，是否还能稳定地重复成功。所谓工程化，不是把事情做复杂，而是把成功变成可复制。

下一篇我们就进到最容易卡人的地方：PC 与 Jetson 双环境版本映射，为什么看似只差一个小版本，结果会差一个世界。

## 配图占位
- [图片占位：版本矩阵表（Jetson/JetPack/CUDA/Python/Paddle）]
- [图片占位：编译流程与报错分层图]
- [图片占位：项目目录与模型路径示意图] -->
