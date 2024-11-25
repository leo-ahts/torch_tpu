在train-sg2260的环境下直接运行python文件

使用chrome浏览器打开chrome://tracing
load使用export_chrome_trace导出的数据，进行可视化分析
或者使用pytorch带的tensorboard进行logs文件的输出，并在命令行中输入以下命令：
tensorboard --logdir=./logs（相对路径）

效果就是分析了简单的卷积神经网络模型在cpu或gpu上的执行情况（包括前向传播、损失计算、反向传播和优化步骤），
记录每个操作的输入和输出tensor的形状，
记录操作过程中每个 Tensor 的内存消耗，
记录堆栈信息，提供有关每个操作的调用栈。

可视化分析如下图所示：
![Uploading _cgi-bin_mmwebwx-bin_webwxgetmsgimg__&MsgID=3849123584373598304&skey=@crypt_cf2811ed_22e0fe37b6cd01355f00717546572aa4&mmweb_appid=wx_webfilehelper.jpeg…]()
