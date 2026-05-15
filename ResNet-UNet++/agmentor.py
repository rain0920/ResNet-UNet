# 导入数据增强工具
import os
import Augmentor

# 确定原始图像存储路径以及掩码文件存储路径
p = Augmentor.Pipeline(r"D:\WY\deep-learning-master\ResNet-UNet++\DataSet\DataSet\JPEGImages1/")
p.ground_truth(r"D:\WY\deep-learning-master\ResNet-UNet++\DataSet\DataSet\SegmentationClass1/")

# 图像旋转： 按照概率0.8执行，最大左旋角度10，最大右旋角度10
p.rotate(probability=0.8, max_left_rotation=10, max_right_rotation=10)

# 图像左右互换： 按照概率0.5执行
p.flip_left_right(probability=0.5)

# 图像放大缩小： 按照概率0.8执行，面积为原始图0.85倍
p.zoom_random(probability=0.3, percentage_area=0.85)

# 最终扩充的数据样本数
# p.sample(200)

img_dir = r"D:\pythonProject\deep-learning-master\ResNet-UNet++\DataSet\DataSet\JPEGImages1"


def __init__(self, path='./'):
    # save the path
    self.path = path
    if not os.path.exists(path):
        os.mkdir(path)

    # 原始输出图片保存路径
    out_dir = os.path.join(img_dir, 'output')
    # 分两个文件夹重新保存image和mask
    save_dir_img = os.path.join(out_dir, 'img')
    save_dir_mask = os.path.join(out_dir, 'mask')
    if not os.path.exists(save_dir_img):
        os.mkdir(save_dir_img)
    if not os.path.exists(save_dir_mask):
        os.mkdir(save_dir_mask)


p.sample(200)

