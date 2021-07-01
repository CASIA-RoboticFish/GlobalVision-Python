# 机器鱼组-全局视觉软件-Python版

## 1 安装方法

### 1.1 安装大恒驱动

安装文件在install文件夹下：Galaxy_Windows_CN_32bits-64bits_1.12.2106.9011

下载网址：https://www.daheng-imaging.com/Software/index.aspx?nodeid=304

选择：大恒相机-windows-usb3.0：下载：[*Galaxy_Windows_CN_32bits/64bits*](javascript:;) 即可

### 1.2 安装Anaconda+Python+包

安装文件在install文件夹下：Anaconda3-2020.07-Windows-x86_64

安装方法请参考：https://blog.csdn.net/ychgyyn/article/details/82119201

需要安装的python包，包括：

- pip install pyserial
- pip install pyyaml
- pip install opencv-python
- pip install opencv-contrib-python

## 2 运行

在工程目录下，运行mainwindow.py文件即可。

IDE可以使用pycharm，也可以使用vscode，大家可按照习惯使用。

## 3 使用说明

本软件的功能包括：

- 实时检测色标块位置、角度、速度、角速度
- 拍摄照片
- 录制视频
- 调试检测效果
- 串口发送数据等

软件使用方法可参考doc文件夹下的“全局视觉软件说明.mp4”视频。

具体步骤如下：

- 前期准备：
  - 关闭所有日光灯，打开水池旁的补光灯。
  - 在要被识别的鱼上贴上红黄色标，红色靠前，黄色靠后。
  - 相机接在USB3.0口
- 第1步：打开大恒软件，打开相机，开始采集，并调整自动增益和自动白平衡
- 第2步：打开全局视觉上位机软件，开始使用

## 4 软件更新

软件存放在Github上，链接为：https://github.com/CASIA-RoboticFish/GlobalVision-Python

后续可能会继续更新部分功能，大家也可以在此基础上进一步修正

## 5 其他说明

- RFLink通讯协议，是我自己定义的通讯协议，具体协议规则可以和我交流

## 6 联系方式

作者：张鹏飞

邮箱：zhangpengfei2017@ia.ac.cn

## 修改记录

- 2021.07.01：加入了储存视频的新线程，录制帧率由9帧，提高到23帧