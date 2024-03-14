这个项目用来训练细粒度 ，边缘敏感，高清的 端到端分割检测算法并且可以实时运行在 移动设备
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
