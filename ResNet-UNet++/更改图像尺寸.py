from PIL import Image
import os.path
import glob


def convertjpg(jpgfile, outdir, width=512, height=512):
    img = Image.open(jpgfile)
    try:
        new_img = img.resize((width, height), Image.BILINEAR)
        new_img.save(os.path.join(outdir, os.path.basename(jpgfile)))
    except Exception as e:
        print(e)


for jpgfile in glob.glob("D:\\WY\\deep-learning-master\\ResNet-UNet++\\miou_pr_dir\\*.png"):
    convertjpg(jpgfile, "D:\\WY\\deep-learning-master\\ResNet-UNet++\\miou_pr_dir1")

for jpgfile in glob.glob("D:\\WY\\deep-learning-master\\ResNet-UNet++\\miou_pr_dir_mask\\*.png"):
    convertjpg(jpgfile, "D:\\WY\\deep-learning-master\\ResNet-UNet++\\miou_pr_dir_mask1")




