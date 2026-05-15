import matplotlib.pyplot as plt
import os
import numpy as np
plt.switch_backend('Agg')

from pylab import mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei']  # 指定默认字体：解决plot不能显示中文问题
mpl.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


# 1.训练时先新建个列表，然后将loss值调用列表的append方法存入列表中
# 2.例如列表train_recon_loss，Discriminator_loss...，然后利用plot即可画出曲线
# 3.最后将画的图保存成图片，imgpath为自定义的图片保存路径。
# plt.figure(num = 2, figsize=(640,480))


# 将acc.csv中的数据转换为列表
# acc = []
# for line in open('D:/WY/deep-learning-master/log/Res50Unet--原代码/acc.csv', 'r', encoding='utf-8'):
#     acc.append(line.strip())  # 一次读一行，并且内存不会溢出，去除空行或者空格
#
# f1score = list(map(float, acc))
# # print(f1score)
#
imgPath = r'D:/WY/deep-learning-master/log/Loss'
# plt.figure()
# plt.xlim((1, 100))  # x轴的取值范围
# plt.ylim((0, 1))  # y轴的取值范围
# plt.plot(f1score, 'b', label='F1-score')
# plt.ylabel('f1-score')
# plt.xlabel('iter_num')
# my_x_ticks = np.arange(1, 101, 6)
# my_y_ticks = np.arange(0, 1.05, 0.05)
# plt.xticks(my_x_ticks)
# plt.yticks(my_y_ticks)
# plt.legend()
# plt.savefig(os.path.join(imgPath, "f1-score.jpg"))


unet_loss = []
u2net_loss = []
res50u2net_loss = []
segnet_loss = []
fcn_loss = []
deeplabv3_loss = []
rdu2net_loss = []

for line in open(r'D:\WY\deep-learning-master\log\Loss\unet.csv', 'r', encoding='utf-8'):
    unet_loss.append(line.strip())  # 一次读一行，并且内存不会溢出，去除空行或者空格

for line in open(r'D:\WY\deep-learning-master\log\Loss\unet++.csv', 'r', encoding='utf-8'):
    u2net_loss.append(line.strip())  # 一次读一行，并且内存不会溢出，去除空行或者空格

for line in open(r'D:\WY\deep-learning-master\log\Loss\res50unet++.csv', 'r', encoding='utf-8'):
    res50u2net_loss.append(line.strip())  # 一次读一行，并且内存不会溢出，去除空行或者空格

for line in open(r'D:\WY\deep-learning-master\log\Loss\segnet.csv', 'r', encoding='utf-8'):
    segnet_loss.append(line.strip())  # 一次读一行，并且内存不会溢出，去除空行或者空格

for line in open(r'D:\WY\deep-learning-master\log\Loss\fcn.csv', 'r', encoding='utf-8'):
    fcn_loss.append(line.strip())  # 一次读一行，并且内存不会溢出，去除空行或者空格

for line in open(r'D:\WY\deep-learning-master\log\Loss\deeplabv3.csv', 'r', encoding='utf-8'):
    deeplabv3_loss.append(line.strip())  # 一次读一行，并且内存不会溢出，去除空行或者空格

for line in open(r'D:\WY\deep-learning-master\log\Loss\r-dunet++.csv', 'r', encoding='utf-8'):
    rdu2net_loss.append(line.strip())  # 一次读一行，并且内存不会溢出，去除空行或者空格


unet_loss1 = list(map(float, unet_loss))
u2net_loss1 = list(map(float, u2net_loss))
res50u2net_loss1 = list(map(float, res50u2net_loss))
segnet_loss1 = list(map(float, segnet_loss))
fcn_loss1 = list(map(float, fcn_loss))
deeplabv3_loss1 = list(map(float, deeplabv3_loss))
rdu2net_loss1 = list(map(float, rdu2net_loss))

plt.figure()
# plt.title()
plt.ylabel('损失值')
plt.xlabel('迭代次数')
plt.xlim(1, 100)  # 设置x轴的范围为[0,1]
plt.ylim(0, 1)  # 同上
my_x_ticks = np.arange(1, 101, 6)  # 设置x轴刻度
my_y_ticks = np.arange(0, 1.05, 0.05)  # 设置y轴刻度
plt.xticks(my_x_ticks)
plt.yticks(my_y_ticks)
plt.plot(unet_loss1, 'b', label='UNet')
plt.plot(u2net_loss1, 'g', label='UNet++')
plt.plot(res50u2net_loss1, 'r', label='ResUNet++')
plt.plot(segnet_loss1, 'c', label='SegNet')
plt.plot(fcn_loss1, 'm', label='FCN')
plt.plot(deeplabv3_loss1, 'y', label='DeepLabv3')
plt.plot(rdu2net_loss1, 'k', label='R-DUnet++')

plt.legend()
plt.savefig(os.path.join(imgPath, "loss.jpg"))




