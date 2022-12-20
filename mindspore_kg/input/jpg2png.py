import os
import PIL.Image as Image


def changeJpgToPng(srcPath, dstPath):
    # 修改图像大小
    image = Image.open(srcPath)

    # 将jpg转换为png
    png_name = str(dstPath)[0:-len('.jpg')] + '.png'
    # image.save(png_name)
    # print(png_name)

    # image = image.convert('RGBA')
    image = image.convert('RGB')
    image.save(png_name)
    pass


if __name__ == '__main__':
    listPath = './'
    srcPath = './'
    dstPath = './'

    print("开始转换...")
    filename_list = os.listdir(listPath)
    for d in filename_list:
        if d.count('.jpg') > 0:
            changeJpgToPng(srcPath + d, dstPath + d)
        pass
    print("完成了...")
