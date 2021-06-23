import cv2
import numpy as np


def kmeans(img):
    # kmeans聚类
    Z = np.float32(img.reshape((-1, 3)))  # 把图像拉伸成一列
    K = 4 # 分成10类
    criteria = (cv2.TERM_CRITERIA_EPS +
                cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    ret, label, center = cv2.kmeans(
        Z, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    center = np.uint8(center) # 这是一个K行3列的矩阵，也就是各个类的中心RGB值
    imgvec = center[label.flatten()]
    img_kmeans = imgvec.reshape((img.shape))  # 获得聚类的图像
    cv2.imshow('kmeans', img_kmeans)
    # # 接下去这一段是选择是red的那一类
    # center_zero = np.float32(center)
    # red_max = (center_zero[0][2] - (center_zero[0][0] + center_zero[0][1]) / 2.0)
    # print(red_max)
    # idmax = 0
    # for i in range(1, K):
    #     print(center_zero[i][2] - (center_zero[i]
    #                                 [0] + center_zero[i][1]) / 2.0)
    #     if (center_zero[i][2] - (center_zero[i][0] + center_zero[i][1]) / 2.0) > red_max:
    #         red_max = (center_zero[i][
    #                     2] - (center_zero[i][0] + center_zero[i][1]) / 2.0)
    #         # print red_max
    #         idmax = i
    # zero = np.zeros(np.shape(center))
    # zero[idmax] = center[idmax]
    # # zero = np.uint8(zero) #如果不转化回uint8，图像显示为黑白色。
    # print("选取第[{}]类为鱼".format(idmax + 1))
    # print("坐标值为")
    # print(zero)
    # img2 = zero[label.flatten()]
    # img2_kmeans = img2.reshape((img.shape))
    return img_kmeans

def detect_ellipse(hsvimg, color_low, color_high):
    thres = cv2.inRange(hsvimg, color_low, color_high)
    median_filter = cv2.medianBlur(thres, 7)  # 中值滤波
    kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    mask = cv2.morphologyEx(median_filter, cv2.MORPH_CLOSE, kernal)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
    results = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    contours = results[1]
    
    ellipse_in_image = None
    for cnt in contours:
        if cnt.size < 10: # 轮廓中点的数量
            print('cnt size')
            print(cnt.size)
            continue
        cnt_area = cv2.contourArea(cnt)
        if cnt_area < 20: # 轮廓围成的点的面积
            print('cnt area')
            print(cnt_area)
            continue
        # # 拟合圆形
        # (x,y),radius = cv2.minEnclosingCircle(cnt)
        # center = (int(x),int(y))
        # radius = int(radius)
        # img = cv2.circle(img,center,radius,(0,255,0),2)
        # # 拟合长方形
        # x, y, w, h = cv2.boundingRect(cnt)
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        # 拟合椭圆
        ellipse = cv2.fitEllipse(cnt)
        ellipse_size =  ellipse[1]
        # if ellipse_size[1]/ellipse_size[0] < 0.8 or ellipse_size[1]/ellipse_size[0] > 2:
        #     print(ellipse_size[1]/ellipse_size[0])
        #     continue
        return ellipse
    return None

def recontruct_point(image_point, depth, K, D):
    undistorted_point = cv2.undistortPoints(image_point, K, D)
    world_point = undistorted_point * depth
    world_point = np.reshape(world_point, (2,))
    return world_point

def undistort_image(distorted_image, K, D):
    undistorted_image = cv2.undistort(distorted_image, K, D)
    cv2.imshow('distort', distorted_image)
    cv2.imshow('undistort', undistorted_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


 
class Tracker(object):
    '''
    追踪者模块,用于追踪指定目标
    '''
 
    def __init__(self, tracker_type="KCF", draw_coord=True):
        '''
        初始化追踪器种类
        '''
        # 获得opencv版本
        (major_ver, minor_ver, subminor_ver) = (cv2.__version__).split('.')
        self.tracker_types = ['BOOSTING', 'MIL', 'KCF', 'TLD', 'MEDIANFLOW', 'GOTURN']
        self.tracker_type = tracker_type
        self.isWorking = False
        self.draw_coord = draw_coord
        # 构造追踪器
        if int(major_ver) < 3:
            self.tracker = cv2.Tracker_create(tracker_type)
        else:
            if tracker_type == 'BOOSTING':
                self.tracker = cv2.TrackerBoosting_create()
            if tracker_type == 'MIL':
                self.tracker = cv2.TrackerMIL_create()
            if tracker_type == 'KCF':
                self.tracker = cv2.TrackerKCF_create()
            if tracker_type == 'TLD':
                self.tracker = cv2.TrackerTLD_create()
            if tracker_type == 'MEDIANFLOW':
                self.tracker = cv2.TrackerMedianFlow_create()
            if tracker_type == 'GOTURN':
                self.tracker = cv2.TrackerGOTURN_create()
 
    def initWorking(self, frame, box):
        '''
        追踪器工作初始化
        frame:初始化追踪画面
        box:追踪的区域
        '''
        if not self.tracker:
            raise Exception("追踪器未初始化")
        status = self.tracker.init(frame, box)
        if not status:
            raise Exception("追踪器工作初始化失败")
        self.coord = box
        self.isWorking = True
 
    def track(self, frame):
        '''
        开启追踪
        '''
        if self.isWorking:
            status, self.coord = self.tracker.update(frame)
            if status:
                if self.draw_coord:
                    # p1 = (int(self.coord[0]), int(self.coord[1]))
                    # p2 = (int(self.coord[0] + self.coord[2]), int(self.coord[1] + self.coord[3]))
                    # cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
                    return self.coord
        return None