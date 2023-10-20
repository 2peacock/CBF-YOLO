# import numpy as np
# import cv2
import random
import os
import cv2
import numpy as np
# import pandas as pd



current_dir = os.path.dirname(__file__)
image_load_dir = os.path.join(current_dir, 'VOCdevkit', 'images', 'train')
label_load_dir = os.path.join(current_dir, 'VOCdevkit', 'labels', 'train')

augimage_save_dir = os.path.join(current_dir, 'VOCdevkitaug', 'image')
auglabel_save_dir = os.path.join(current_dir, 'VOCdevkitaug', 'label')

label_files = [f for f in os.listdir(label_load_dir) if f.endswith('.txt')]  ##返回值是列表

for f in label_files:
    name = f.split(".")[0]
    # 读取图片
    img = cv2.imread(os.path.join(image_load_dir, name + ".jpg"))

    # p = 0.5
    # # 模拟一次试验
    # n = 1
    # 使用binomial函数进行概率判断
    # successes = np.random.binomial(n, p)
    successes=1
    # 判断是否满足条件
    if successes > 0:
        # 获取图像尺寸和通道数
        h, w, c = img.shape
        angle = random.randint(0, 30)
        #线条间隔和长度
        rand_x = random.randint(100, 125)
        rand_y = random.randint(2, 5)


        # 创建一个全白的1000x1000的幕布
        white_canvas = np.full((1000, 1000, 3), (255, 255, 255), dtype=np.uint8)
        # 水平方向的线条
        for y in range(0, 1000, rand_x):
            cv2.line(white_canvas, (0, y), (1000, y), (0, 0, 0), rand_y)


        # 垂直方向的线条
        for x in range(0, 1000, rand_x):
            cv2.line(white_canvas, (x, 0), (x, 1000), (0, 0, 0), rand_y)

        # 水平方向旋转矩阵
        M = cv2.getRotationMatrix2D((white_canvas.shape[1] / 2, white_canvas.shape[0] / 2), angle, 1)
        # 仿射变换
        rotated_canvas = cv2.warpAffine(white_canvas, M, (white_canvas.shape[1], white_canvas.shape[0]))

        # 中心点坐标
        cx, cy = white_canvas.shape[1] // 2, white_canvas.shape[0] // 2

        # 截取部分图像素坐标（左上角为原点）
        cropped_canvas = rotated_canvas[cy - 250:cy + 250, cx - 250:cx + 250]

        masked_img = cv2.bitwise_and(img, cropped_canvas)

        # 显示掩码运算后的图片
        # cv2.imshow('Masked Image', masked_img)

        # 显示截取的部分图
        # cv2.imshow('Cropped Canvas', cropped_canvas)

        # 显示幕布

        # cv2.waitKey(0)
        # cv2.destroyAllWindows()
        cv2.imwrite(os.path.join(augimage_save_dir, name + '_aug.jpg'), masked_img)
        print("完成" + name)

    else:
        continue





