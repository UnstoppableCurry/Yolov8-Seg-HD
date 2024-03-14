这个项目用来训练细粒度 ，边缘敏感，高清的 端到端分割检测算法并且可以实时运行在 移动设备
并且api用法一切都与原u版本对齐，不过u版最新的一些特性是不支持的 例如可视化 旋转 最终  视为与2023 版本的ultralytics 一致
鉴于ultralytics 致力于API更简单使用，但是对于开发者而言 并不能想torch框架一样任意修改结构 与修改需求 所以 就做了一个满足我需求的 API不友好 开发者友好的yolov8-seg
# data
  train
    images
      1.jpg,2.jpg
    labels
      1.txt,2.txt
    label
      1.png,2.png
  val
    images
      1.jpg,2.jpg
    labels
      1.txt,2.txt
    label
      1.png,2.png

# seg train
label中的数据必须与labels中的数据对应
txt 的轮廓 其实就是 二值图上绘制的白色区域
