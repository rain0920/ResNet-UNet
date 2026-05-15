import numpy as np
import cv2
import matplotlib.pyplot as plt
image = cv2.imread(r'D:\pythonProject\deep-learning-master\ResNet-UNet++\DataSet\DataSet\JPEGImages\000332.png', cv2.IMREAD_COLOR)
img1 = np.array(image, dtype='int')  # 转换成int型，不然会导致数据溢出
# 超绿灰度图
b, g, r = cv2.split(img1)
# ExG_sub = cv2.subtract(2*g,r)
# ExG = cv2.subtract(ExG_sub,b )
# ExG = 2 * g - r - b
# ExG = 0.441*r-0.811*g+0.385*b+18.78745
ExG = 1.4*r-g
[m, n] = ExG.shape

for i in range(m):
    for j in range(n):
        if ExG[i, j] < 0:
            ExG[i, j] = 0
        elif ExG[i, j] > 255:
            ExG[i, j] = 255

ExG = np.array(ExG, dtype='uint8')  # 重新转换成uint8类型
ret2, th2 = cv2.threshold(ExG, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

plt.figure(figsize=(20, 10), dpi=200)
plt.subplot(131), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title('Original'), plt.axis('off')
plt.subplot(132), plt.imshow(cv2.cvtColor(ExG, cv2.COLOR_BGR2RGB)), plt.title('ExG_gray'), plt.axis('off')
plt.subplot(133), plt.imshow(cv2.cvtColor(th2, cv2.COLOR_BGR2RGB)), plt.title('OTSU_bw'), plt.axis('off')

plt.savefig('D:/pythonProject/deep-learning-master/ResNet-UNet++/exgimages/OTSU_bw.png')
plt.show()
