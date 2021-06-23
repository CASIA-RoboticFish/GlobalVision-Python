
import cv2
import numpy as np
import imageprocess
import sys


def find_fish_kmeans(image):
    kernel = np.ones((5, 5), np.uint8)
    img = image.copy()
    if image is None:
        print("图像加载错误")
        sys.exit()
    else:
        if len(np.shape(image)) != 3:
            print("不是多通道图片，请确认图像类型")
            sys.exit()
        else:
            m, n, _ = np.shape(img)
            # kmeans聚类
            Z = np.float32(img.reshape((-1, 3)))  # 把图像拉伸成一列
            print("Z的尺寸为[{}]".format(np.shape(Z)))
            criteria = (cv2.TERM_CRITERIA_EPS +
                        cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
            K = 10
            ret, label, center = cv2.kmeans(
                Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            center = np.uint8(center) # 这是一个K行3列的矩阵，也就是各个类的中心RGB值
            print("中心点坐标为")
            print(center)
            print("label的尺寸为[{}]".format(np.shape(label.flatten())))
            img1 = center[label.flatten()]
            img1_kmeans = img1.reshape((img.shape))  # 获得聚类的图像
            # 接下去这一段是选择是red的那一类
            center_zero = np.float32(center)
            red_max = (center_zero[0][2] - (center_zero[0][0] + center_zero[0][1]) / 2.0)
            print(red_max)
            idmax = 0
            for i in range(1, K):
                print(center_zero[i][2] - (center_zero[i]
                                           [0] + center_zero[i][1]) / 2.0)
                if (center_zero[i][2] - (center_zero[i][0] + center_zero[i][1]) / 2.0) > red_max:
                    red_max = (center_zero[i][
                               2] - (center_zero[i][0] + center_zero[i][1]) / 2.0)
                    # print red_max
                    idmax = i
            zero = np.zeros(np.shape(center))
            zero[idmax] = center[idmax]
           # zero = np.uint8(zero) #如果不转化回uint8，图像显示为黑白色。
            print("选取第[{}]类为鱼".format(idmax + 1))
            print("坐标值为")
            print(zero)
            img2 = zero[label.flatten()]
            img2_kmeans = img2.reshape((img.shape))
            return img1_kmeans, img2_kmeans, center, label

if __name__ == '__main__':
    cap = cv2.VideoCapture(0)
    # low_hue = self.hue_low_splider.value()
    # high_hue = self.hue_high_splider.value()
    # low_sat = self.satur_low_splider.value()
    # high_sat = self.satur_high_splider.value()
    # low_val = self.value_low_splider.value()
    # high_val = self.value_high_splider.value()
    # color_low = np.array([low_hue, low_sat, low_val])
    # color_high = np.array([high_hue, high_sat, high_val])

    while(cap.isOpened()):
        ret, frame = cap.read()
        if ret:
            # 图像预处理
            blur = cv2.GaussianBlur(frame, (7, 7), 0)
            hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
            imageprocess.kmeans(blur)
            cv2.imshow('frame', frame)
            cv2.waitKey(10)
    
    cap.release()
    cv2.destroyAllWindows()
            