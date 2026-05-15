import cv2
import numpy as np
import matplotlib.pyplot as plt

imgfile = r'D:\wy\deep-learning-master\ResNet-UNet++\DataSet\DataSet\JPEGImages\000338.png'
pngfile = r'D:\wy\deep-learning-master\ResNet-UNet++\miou_pr_dir\000338.png'

img = cv2.imread(imgfile, 1)
mask = cv2.imread(pngfile, 0)

contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0, 0, 255), 1)

img = img[:, :, ::-1]
img[..., 2] = np.where(mask == 1, 255, img[..., 2])

cv2.imwrite("img_out/00001.jpg", img)
# plt.imshow(img)
plt.show()














