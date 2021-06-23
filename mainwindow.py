import sys
import time
import math
import datetime
import serial
import struct
import copy
import platform
import yaml
import cv2
from PyQt5 import QtCore,QtGui,QtWidgets
import numpy as np

import gxipy as gx
import imageprocess
import serctl # 串口控制工具
import rflink # Robotic Fish 通讯协议

## 这个类帮助在图片中绘制矩形，给KCF算法作为初始ROI
class ImageLabel(QtWidgets.QLabel):
    x0 = 0
    y0 = 0
    x1 = 0
    y1 = 0
    flag = False
    #鼠标点击事件
    def mousePressEvent(self,event):
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()
    #鼠标释放事件
    def mouseReleaseEvent(self,event):
        self.flag = False
    #鼠标移动事件
    def mouseMoveEvent(self,event):
        if self.flag:
            self.x1 = event.x()
            self.y1 = event.y()
            self.update()
    #绘制事件
    def paintEvent(self, event):
        super().paintEvent(event)
        rect =QtCore.QRect(self.x0, self.y0, abs(self.x1-self.x0), abs(self.y1-self.y0))
        painter = QtGui.QPainter(self)
        painter.setPen(QtGui.QPen(QtCore.Qt.red,2,QtCore.Qt.SolidLine))
        painter.drawRect(rect)
        # print(self.x0, self.y0, self.x1, self.y1)
    
    def getRect(self):
        return self.x0, self.y0, self.x1, self.y1

    def clearRect(self):
        self.x0 = 0
        self.y0 = 0
        self.x1 = 0
        self.y1 = 0


## 主窗口
class GLOBALVISION(QtWidgets.QMainWindow): # 主窗口

    close_signal = QtCore.pyqtSignal() # 同步关闭主窗口和子窗口
    
    # 初始化
    def __init__(self):
        super(GLOBALVISION, self).__init__()
        # 初始化串口
        self.serialtool = serctl.Serial()
        # 初始化RFLink
        self.rftool = rflink.RFLink()
        # 初始化跟踪器
        self.tracker = imageprocess.Tracker(tracker_type="KCF")
        self.kcfroi = None
        self.roi_reset_cnt = 0
        # 读取HSV参数
        self.paramfilepath = 'config/param.yml'
        paramfile = open(self.paramfilepath, 'r', encoding='utf-8')
        paramdata = paramfile.read()
        self.hsvparam = yaml.load(paramdata)

        # 读取背景图像
        self.backgroundpath = 'background.jpg'
        try:
            self.background = cv2.imread(self.backgroundpath, cv2.IMREAD_GRAYSCALE)
        except:
            self.background = None

        # 读取Camera参数
        self.camerafilepath = 'config/camera.yml'
        camerafile = open(self.camerafilepath, 'r', encoding='utf-8')
        camerainfo = camerafile.read()
        camera_param = yaml.load(camerainfo)
        self.cam_K = np.array(camera_param['Camera']['K'])
        self.cam_D = np.array(camera_param['Camera']['D'])
        self.cam_border =  np.array(camera_param['Camera']['border']).reshape((4,))
        self.cam_width = camera_param['Camera']['width']
        self.cam_height = camera_param['Camera']['height']
        self.target_depth = camera_param['Camera']['depth']
        self.roi_reset_num = camera_param['Detection']['roi_reset_num']
        self.cam_exposure = camera_param['Camera']['exposure']
        # 视频变量
        self.image = None
        self.video_frame_rate = 20
        self.video_frame_interval = 50 # 1000/self.video_frame_rate
        self.cam = None
        self.cam_device_manager = None
        self.cam_gamma_lut = None
        self.cam_contrast_lut = None
        self.cam_color_correction_param = 0
        self.videowriter = None
        self.filewriter = None
        self.camera_start_time = 0
        self.camera_time = 0

        # 状态Flag
        self.camera_flag = False
        self.detection_flag = False
        self.recordvideo_flag = False
        self.debugmode_flag = False
        self.control_flag = False
        self.parameter_flag = False
        self.serialport_flag = False
        self.detect_success_flag = False

        # 显示Flag
        self.time_show_flag = True
        self.border_show_flag = False
        self.cameraparam_show_flag = False
        self.detectstamp_show_flag = False
        self.background_sub_flag = False
        self.tracking_mode_flag = False

        # 速度计算
        self.time_interval = 0
        self.pre_time = 0
        self.now_time = time.time()
        self.pre_position = np.zeros((2,))
        self.now_position = np.zeros((2,))
        self.pre_velocity = np.zeros((2,))
        self.now_velocity = np.zeros((2,))
        self.now_velocity_norm = 0
        self.pre_angle = 0
        self.now_angle = 0
        self.pre_angvel = 0
        self.now_angvel = 0
        self.max_velocity_norm = 0
        self.max_angvel = 0
        self.point_1 = np.zeros((2,))
        self.point_2 = np.zeros((2,))
        self.point_3 = np.zeros((2,))
        self.turn_radius_cnt = 0
        self.turn_radius = 0
        self.average_turn_radius = 0
        self.min_turn_radius = 0
        self.red_point_img_pos_x = 0
        self.red_point_img_pos_y = 0
        self.yellow_point_img_pos_x = 0
        self.yellow_point_img_pos_y = 0
        # 初始化UI
        self.init_ui()
        self.widget_connect()
       
    # 初始化UI界面
    def init_ui(self):
        """
        初始化UI
        :return:
        """
        self.init_layout()
        self.statusBar().showMessage('欢迎使用全局视觉测量软件')
        self.setFixedSize(1500, 1000)# 设置窗体大小
        self.setWindowTitle('全局视觉测量软件')  # 设置窗口标题
        self.show()  # 窗口显示

    # 初始化layout界面
    def init_layout(self):
        """
        初始化UI界面布局
        :return:
        """
        # 定时器
        self.timer = QtCore.QTimer()
        self.timer.setInterval(self.video_frame_interval)
        
        # 布局
        self.main_widget = QtWidgets.QWidget()
        self.main_layout = QtWidgets.QGridLayout()
        self.main_widget.setLayout(self.main_layout)

        # 图片控件
        self.image_window = ImageLabel(self)
        self.image_window.setFixedSize(1292,964)
        self.image_window.setCursor(QtCore.Qt.CrossCursor)

        # 控制按钮
        self.control_button_label = QtWidgets.QLabel('控制按钮')
        self.control_button_label.setFont(QtGui.QFont('Microsoft YaHei', 10, QtGui.QFont.Bold))

        self.open_camera_button = QtWidgets.QPushButton('打开相机')
        # self.open_camera_button.setFixedSize(90,25)
        self.close_camera_button = QtWidgets.QPushButton('关闭相机')
        # self.close_camera_button.setFixedSize(90,25)
        self.enable_detect_button = QtWidgets.QPushButton('开始检测')
        # self.enable_detect_button.setFixedSize(90,25)
        self.disable_detect_button = QtWidgets.QPushButton('停止检测')
        # self.disable_detect_button.setFixedSize(90,25)
        self.take_photo_button = QtWidgets.QPushButton('拍摄照片')
        # self.take_photo_button.setFixedSize(90,25)
        self.record_video_button = QtWidgets.QPushButton('录制视频')
        # self.record_video_button.setFixedSize(90,25)

        # 显示控制
        self.display_control_label = QtWidgets.QLabel('显示控制')
        self.display_control_label.setFont(QtGui.QFont('Microsoft YaHei', 10, QtGui.QFont.Bold))
        self.time_checkbox = QtWidgets.QCheckBox("时间")
        self.time_checkbox.setChecked(True)
        self.border_checkbox = QtWidgets.QCheckBox("边界框")
        self.border_checkbox.setChecked(False)
        self.cameraparam_checkbox = QtWidgets.QCheckBox("相机参数")
        self.cameraparam_checkbox.setChecked(False)
        self.detectstamp_checkbox = QtWidgets.QCheckBox("检测记号")
        self.detectstamp_checkbox.setChecked(False)
        self.background_checkbox = QtWidgets.QCheckBox("背景减除")
        self.background_checkbox.setChecked(False)

        # 参数调节
        self.param_adjust_label = QtWidgets.QLabel('参数调节')
        self.param_adjust_label.setFont(QtGui.QFont('Microsoft YaHei', 10, QtGui.QFont.Bold))
        self.red_marker_checkbox = QtWidgets.QCheckBox("红色标记")
        self.yellow_marker_checkbox = QtWidgets.QCheckBox("黄色标记")
        self.red_marker_checkbox.setChecked(True)

        self.hue_low_label = QtWidgets.QLabel('色相低')
        self.hue_low_splider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.hue_high_label = QtWidgets.QLabel('色相高')
        self.hue_high_splider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.satur_low_label = QtWidgets.QLabel('饱和低')
        self.satur_low_splider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.satur_high_label = QtWidgets.QLabel('饱和高')
        self.satur_high_splider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.value_low_label = QtWidgets.QLabel('明度低')
        self.value_low_splider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.value_high_label = QtWidgets.QLabel('明度高')
        self.value_high_splider = QtWidgets.QSlider(QtCore.Qt.Horizontal)
        self.hue_low_splider.setMinimum(0)
        self.hue_low_splider.setMaximum(255)
        self.hue_high_splider.setMinimum(0)
        self.hue_high_splider.setMaximum(255)
        self.satur_low_splider.setMinimum(0)
        self.satur_low_splider.setMaximum(255)
        self.satur_high_splider.setMinimum(0)
        self.satur_high_splider.setMaximum(255)
        self.value_low_splider.setMinimum(0)
        self.value_low_splider.setMaximum(255)
        self.value_high_splider.setMinimum(0)
        self.value_high_splider.setMaximum(255)

        self.refresh_hsvparam_ui()

        self.debug_mode_button = QtWidgets.QPushButton('调试模式')
        # self.debug_mode_button.setFixedSize(90,25)
        self.save_parameter_button = QtWidgets.QPushButton('保存参数')
        # self.save_parameter_button.setFixedSize(90,25)
        self.save_background_button = QtWidgets.QPushButton('拍摄背景')
        # self.save_parameter_button.setFixedSize(90,25)
        self.set_kcfroi_button = QtWidgets.QPushButton('设置跟踪对象')
        # self.save_parameter_button.setFixedSize(90,25)

        # 串口控制
        self.serial_control_label = QtWidgets.QLabel('串口控制')
        self.serial_control_label.setFont(QtGui.QFont('Microsoft YaHei', 10, QtGui.QFont.Bold))
        self.serial_com_label = QtWidgets.QLabel('COM')
        self.serial_com_combo = QtWidgets.QComboBox()
        self.serial_com_combo.addItem('COM3')
        self.serial_com_combo.addItem('COM5')
        self.serial_com_combo.addItem('COM6')
        self.serial_com_combo.addItem('COM7')
        self.serial_com_combo.addItem('COM9')
        self.serial_com_combo.addItem('COM10')
        self.serial_com_combo.addItem('COM11')
        self.serial_com_combo.addItem('COM12')
        self.serial_com_combo.addItem('COM13')
        self.serial_com_combo.addItem('COM28')

        self.serial_bps_label = QtWidgets.QLabel('BPS')
        self.serial_bps_combo = QtWidgets.QComboBox()
        self.serial_bps_combo.addItem('9600')
        self.serial_bps_combo.addItem('28400')
        self.serial_bps_combo.addItem('19200')
        self.serial_bps_combo.addItem('38400')
        self.serial_bps_combo.addItem('56000')
        self.serial_bps_combo.addItem('57600')
        self.serial_bps_combo.addItem('129200')

        self.serial_open_button = QtWidgets.QPushButton('开始发送')
        # self.serial_open_button.setFixedSize(90,25)
        self.serial_close_button = QtWidgets.QPushButton('停止发送')
        # self.serial_close_button.setFixedSize(90,25)


        # 布局,29行17列
        row_cnt = 0
        self.main_layout.addWidget(self.image_window, 0, 0, 20, 28)
        self.main_layout.addWidget(self.control_button_label, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.open_camera_button, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.close_camera_button, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.enable_detect_button, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.disable_detect_button, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.take_photo_button, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.record_video_button, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.display_control_label, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.time_checkbox, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.border_checkbox, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.cameraparam_checkbox, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.detectstamp_checkbox, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.background_checkbox, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.param_adjust_label, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.red_marker_checkbox, row_cnt, 28, 1, 2)
        self.main_layout.addWidget(self.yellow_marker_checkbox, row_cnt, 30, 1, 2)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.hue_low_label, row_cnt, 28, 1, 1)
        self.main_layout.addWidget(self.hue_low_splider, row_cnt, 29, 1, 3)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.hue_high_label, row_cnt, 28, 1, 1)
        self.main_layout.addWidget(self.hue_high_splider, row_cnt, 29, 1, 3)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.satur_low_label, row_cnt, 28, 1, 1)
        self.main_layout.addWidget(self.satur_low_splider, row_cnt, 29, 1, 3)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.satur_high_label, row_cnt, 28, 1, 1)
        self.main_layout.addWidget(self.satur_high_splider, row_cnt, 29, 1, 3)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.value_low_label, row_cnt, 28, 1, 1)
        self.main_layout.addWidget(self.value_low_splider, row_cnt, 29, 1, 3)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.value_high_label, row_cnt, 28, 1, 1)
        self.main_layout.addWidget(self.value_high_splider, row_cnt, 29, 1, 3)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.debug_mode_button, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.save_parameter_button, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.save_background_button, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.set_kcfroi_button, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.serial_control_label, row_cnt, 28, 1, 4)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.serial_com_label, row_cnt, 28, 1, 1)
        self.main_layout.addWidget(self.serial_com_combo, row_cnt, 29, 1, 3)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.serial_bps_label, row_cnt, 28, 1, 1)
        self.main_layout.addWidget(self.serial_bps_combo, row_cnt, 29, 1, 3)
        row_cnt = row_cnt + 1
        self.main_layout.addWidget(self.serial_open_button, row_cnt, 28, 1, 2)
        self.main_layout.addWidget(self.serial_close_button, row_cnt, 30, 1, 2)

        self.setCentralWidget(self.main_widget)  # 设置窗口主部件

    def widget_connect(self):
        self.timer.timeout.connect(self.play_video)
        self.open_camera_button.clicked.connect(self.open_camera_button_slot)
        self.close_camera_button.clicked.connect(self.close_camera_button_slot)
        self.enable_detect_button.clicked.connect(self.enable_detect_button_slot)
        self.disable_detect_button.clicked.connect(self.disable_detect_button_slot)
        self.time_checkbox.clicked.connect(self.time_checkbox_slot)
        self.border_checkbox.clicked.connect(self.border_checkbox_slot)
        self.cameraparam_checkbox.clicked.connect(self.cameraparam_checkbox_slot)
        self.detectstamp_checkbox.clicked.connect(self.detectstamp_checkbox_slot)
        self.background_checkbox.clicked.connect(self.background_checkbox_slot)
        self.yellow_marker_checkbox.stateChanged.connect(self.yellow_marker_checkbox_slot)
        self.red_marker_checkbox.stateChanged.connect(self.red_marker_checkbox_slot)
        self.debug_mode_button.clicked.connect(self.debug_mode_button_slot)
        self.save_parameter_button.clicked.connect(self.save_parameter_button_slot)
        self.take_photo_button.clicked.connect(self.take_photo_button_slot)
        self.record_video_button.clicked.connect(self.record_video_button_slot)
        self.save_background_button.clicked.connect(self.save_background_button_slot)
        self.set_kcfroi_button.clicked.connect(self.set_kcfroi_button_slot)
        self.serial_open_button.clicked.connect(self.serial_open_button_slot)
        self.serial_close_button.clicked.connect(self.serial_close_button_slot)

    def play_video(self):
        if (self.camera_flag):
            # 获取图像
            self.camera_time = time.time() - self.camera_start_time
            img = self.acq_color()

            if self.time_show_flag:
                self.add_time(img)

            if self.recordvideo_flag:
                bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
                self.videowriter.write(bgr)

            # 背景减除功能
            if self.background_sub_flag:
                img = self.background_sub(img)
            else:
                if self.detection_flag:
                    img = self.detection(img)
                
                if self.border_show_flag:
                    img = self.add_border(img)
                
                if self.cameraparam_show_flag:
                    img = self.add_cameraparam(img)
                
                if self.detect_success_flag:
                    if self.serialport_flag:
                        self.send_rflink_data()
                         
            # 记录数据
            if self.recordvideo_flag:
                datastr = ('%.2f' % self.camera_time) + ' ' + ('%.6f' % self.now_angle) + ' ' + ('%.6f' % self.now_angvel) + ' ' + ('%.6f' % self.now_velocity[0]) + ' ' + ('%.6f' % self.now_velocity[1]) + \
                      ' ' + ('%.6f' % self.now_position[0]) + ' ' + ('%.6f' % self.now_position[1]) + ' ' + ('%.6f' % self.turn_radius)  + ' ' +  ('%.2f' % self.red_point_img_pos_x) + ' ' +  ('%.2f' % self.red_point_img_pos_y) + \
                      ' ' +  ('%.2f' % self.yellow_point_img_pos_x) + ' ' +  ('%.2f' % self.yellow_point_img_pos_y)+ '\n'
                self.filewriter.write(datastr)
            
            # 保存当前帧
            self.image = img

            # 转为QImage对象
            height, width, bytesPerComponent = img.shape
            bytesPerLine = bytesPerComponent * width
            qtimage = QtGui.QImage(img.data.tobytes(), width, height, bytesPerLine, QtGui.QImage.Format_RGB888)
            self.image_window.setPixmap(QtGui.QPixmap.fromImage(qtimage).scaled(self.image_window.width(), self.image_window.height()))

            # 恢复detect_success_flag
            self.detect_success_flag = False

    def acq_color(self):
        # send software trigger command
        self.cam.TriggerSoftware.send_command()

        # get raw image
        raw_image = self.cam.data_stream[0].get_image()
        if raw_image is None:
            print("Getting image failed.")
            return None
        # image_ts = raw_image.get_timestamp()
        # image_id = raw_image.get_frame_id()
        # get RGB image from raw image
        rgb_image = raw_image.convert("RGB")
        if rgb_image is None:
            return None

        # improve image quality
        rgb_image.image_improvement(self.cam_color_correction_param, self.cam_contrast_lut, self.cam_gamma_lut)

        # create numpy array with data from raw image
        numpy_image = rgb_image.get_numpy_array()
        if numpy_image is None:
            return None

        # show acquired image
        img = numpy_image
        return img

    def background_sub(self, img):
        if self.background is not None:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            sub = cv2.absdiff(gray, self.background)
            ret, thres = cv2.threshold(sub, 50, 255, cv2.THRESH_BINARY)
            kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            mask = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernal)
            result = cv2.bitwise_and(img, img, mask = mask)
        return result

    def detection(self, img):
        # kcf跟踪
        kcf_mask = np.ones((self.cam_height, self.cam_width), dtype='uint8')
        if self.kcfroi is None:
            pass
        else:
            self.kcfroi = self.tracker.track(img)
            if self.kcfroi is None:
                pass
            else:
                p1 = (int(self.kcfroi[0]), int(self.kcfroi[1]))
                p2 = (int(self.kcfroi[0] + self.kcfroi[2]), int(self.kcfroi[1] + self.kcfroi[3]))
                #if self.detectstamp_show_flag:
                cv2.rectangle(img, p1, p2, (0, 0, 0), 2, 1)
                kcf_mask[:,:]=0
                kcf_mask[p1[1]:p2[1],p1[0]:p2[0]] = 1

        # 图像预处理
        blur = cv2.GaussianBlur(img, (7, 7), 0)
        
        # 是否进入Debug模式
        if self.debugmode_flag:
            low_hue = self.hue_low_splider.value()
            high_hue = self.hue_high_splider.value()
            low_sat = self.satur_low_splider.value()
            high_sat = self.satur_high_splider.value()
            low_val = self.value_low_splider.value()
            high_val = self.value_high_splider.value()
            color_low = np.array([low_hue, low_sat, low_val])
            color_high = np.array([high_hue, high_sat, high_val])
            if self.kcfroi is None:
                hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                thres = cv2.inRange(hsv, color_low, color_high)
                kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernal)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
                mask_border = np.zeros((self.cam_height, self.cam_width), dtype='uint8')
                mask_border[self.cam_border[2]:self.cam_border[3],self.cam_border[0]:self.cam_border[1]] = mask[self.cam_border[2]:self.cam_border[3],self.cam_border[0]:self.cam_border[1]]
                result = cv2.bitwise_and(img, img, mask = mask_border)
            else:
                p1 = (int(self.kcfroi[0]), int(self.kcfroi[1]))
                p2 = (int(self.kcfroi[0] + self.kcfroi[2]), int(self.kcfroi[1] + self.kcfroi[3]))
                # 先用Kmeans分割图像
                roi = blur[p1[1]:p2[1],p1[0]:p2[0],:]
                kmeansroi = imageprocess.kmeans(roi)
                # 再用HSV分割颜色
                kmeansroihsv = cv2.cvtColor(kmeansroi, cv2.COLOR_BGR2HSV)
                thres = cv2.inRange(kmeansroihsv, color_low, color_high)
                # 形态学操作
                kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
                mask = cv2.morphologyEx(thres, cv2.MORPH_CLOSE, kernal)
                mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal)
                # 图像位与操作
                mask_border = np.zeros((self.cam_height, self.cam_width), dtype='uint8')
                mask_border[p1[1]:p2[1],p1[0]:p2[0]] = mask
                result = cv2.bitwise_and(img, img, mask = mask_border)
            return result

        # 检测标记
        # 红色
        red_low_hue = self.hsvparam['RedTarget']['Hmin']
        red_high_hue = self.hsvparam['RedTarget']['Hmax']
        red_low_sat = self.hsvparam['RedTarget']['Smin']
        red_high_sat = self.hsvparam['RedTarget']['Smax']
        red_low_val = self.hsvparam['RedTarget']['Vmin']
        red_high_val = self.hsvparam['RedTarget']['Vmax']
        red_color_low = np.array([red_low_hue, red_low_sat, red_low_val])
        red_color_high = np.array([red_high_hue, red_high_sat, red_high_val])
        red_x = 0
        red_y = 0
        # 黄色red_
        yellow_low_hue = self.hsvparam['YellowTarget']['Hmin']
        yellow_high_hue = self.hsvparam['YellowTarget']['Hmax']
        yellow_low_sat = self.hsvparam['YellowTarget']['Smin']
        yellow_high_sat = self.hsvparam['YellowTarget']['Smax']
        yellow_low_val = self.hsvparam['YellowTarget']['Vmin']
        yellow_high_val = self.hsvparam['YellowTarget']['Vmax']
        yellow_color_low = np.array([yellow_low_hue, yellow_low_sat, yellow_low_val])
        yellow_color_high = np.array([yellow_high_hue, yellow_high_sat, yellow_high_val])
        yellow_x = 0
        yellow_y = 0 
        try:
            if self.kcfroi is None:
                hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
                red_ellipse = imageprocess.detect_ellipse(hsv[self.cam_border[2]:self.cam_border[3], self.cam_border[0]:self.cam_border[1],:], red_color_low, red_color_high)
                if red_ellipse is not None:
                    ellipse_center = red_ellipse[0]
                    red_x = round(ellipse_center[0]) + self.cam_border[0]
                    red_y = round(ellipse_center[1]) + self.cam_border[2]
                yellow_ellipse = imageprocess.detect_ellipse(hsv[self.cam_border[2]:self.cam_border[3], self.cam_border[0]:self.cam_border[1],:], yellow_color_low, yellow_color_high)
                if yellow_ellipse is not None:
                    ellipse_center = yellow_ellipse[0]
                    yellow_x = round(ellipse_center[0]) + self.cam_border[0]
                    yellow_y = round(ellipse_center[1]) + self.cam_border[2]
            else:
                p1 = (int(self.kcfroi[0]), int(self.kcfroi[1]))
                p2 = (int(self.kcfroi[0] + self.kcfroi[2]), int(self.kcfroi[1] + self.kcfroi[3]))
                # 先用Kmeans分割图像
                roi = blur[p1[1]:p2[1],p1[0]:p2[0],:]
                kmeansroi = imageprocess.kmeans(roi)
                # 再用HSV分割颜色
                kmeansroihsv = cv2.cvtColor(kmeansroi, cv2.COLOR_BGR2HSV)
                red_ellipse = imageprocess.detect_ellipse(kmeansroihsv, red_color_low, red_color_high)
                if red_ellipse is not None:
                    ellipse_center = red_ellipse[0]
                    red_x = round(ellipse_center[0]) + int(self.kcfroi[0])
                    red_y = round(ellipse_center[1]) + int(self.kcfroi[1])
                yellow_ellipse = imageprocess.detect_ellipse(kmeansroihsv, yellow_color_low, yellow_color_high)
                if yellow_ellipse is not None:
                    ellipse_center = yellow_ellipse[0]
                    yellow_x = round(ellipse_center[0]) + int(self.kcfroi[0])
                    yellow_y = round(ellipse_center[1]) + int(self.kcfroi[1])
        except:
            red_ellipse = None
            yellow_ellipse = None

        # 计算色标三维位置
        red_point_pos = None
        yellow_point_pos = None
        if red_ellipse is not None and yellow_ellipse is not None:
            # Todo:对色标位置加入卡尔曼滤波
            red_point = np.zeros((1,1,2))
            # red_point = red_point.reshape(1, 1, 2) # 点必须是三维的（1,1,2)，第一维是点的个数
            red_point[0,0,0] = red_x
            red_point[0,0,1] = red_y
            red_point_pos = imageprocess.recontruct_point(red_point, self.target_depth, self.cam_K, self.cam_D)
            self.red_point_img_pos_x = red_point[0,0,0]
            self.red_point_img_pos_y = red_point[0,0,1]
        
            yellow_point = np.zeros((1,1,2))
            # yellow_point = yellow_point.reshape(1, 1, 2) # 点必须是三维的（1,1,2)，第一维是点的个数
            yellow_point[0,0,0] = yellow_x
            yellow_point[0,0,1] = yellow_y
            yellow_point_pos = imageprocess.recontruct_point(yellow_point, self.target_depth, self.cam_K, self.cam_D)
            self.yellow_point_img_pos_x = yellow_point[0,0,0]
            self.yellow_point_img_pos_y = yellow_point[0,0,1]
            
        # 计算速度
        if red_point_pos is not None and yellow_point_pos is not None:
            if self.pre_time == 0:
                self.pre_time = time.time()
                self.now_time = time.time()
                self.pre_position = (red_point_pos[:2] + yellow_point_pos[:2])/2/1000 # 单位：米
                self.now_position = self.pre_position
                self.pre_velocity = np.zeros((2,))
                self.now_velocity = self.pre_velocity

                vec_r2y = yellow_point_pos[:2] - red_point_pos[:2]
                self.pre_angle = math.atan2(vec_r2y[1], vec_r2y[0]) # 单位：弧度
                self.pre_angle = self.pre_angle
                self.pre_angvel = 0
                self.now_angvel = self.pre_angvel

                self.point_1 = self.pre_position
                self.point_2 = self.pre_position
                self.point_3 = self.pre_position
            else:
                # 时间
                self.pre_time = self.now_time
                self.now_time = time.time()
                time_interval = self.now_time - self.pre_time # 单位：秒
                self.time_interval = time_interval
                # 速度
                alpha = 0.8
                beta = 1 - alpha
                self.now_position = (red_point_pos[:2] + yellow_point_pos[:2])/2 # 单位：米
                self.now_velocity = (self.now_position - self.pre_position) / time_interval
                self.now_velocity = self.pre_velocity * alpha + self.now_velocity * beta
                self.pre_position = self.now_position
                self.pre_velocity = self.now_velocity
                self.now_velocity_norm = np.linalg.norm(self.now_velocity)
                if self.now_velocity_norm > self.max_velocity_norm:
                    self.max_velocity_norm = self.now_velocity_norm
                # 角速度
                # vec_r2y = yellow_point_pos[:2] - red_point_pos[:2]
                vec_r2y = red_point_pos[:2] - yellow_point_pos[:2]
                self.now_angle = math.atan2(vec_r2y[1], vec_r2y[0])
                angle_interval = self.now_angle - self.pre_angle
                if abs(angle_interval) > 3*3.14/2:
                    self.now_angvel = self.now_angvel
                else:
                    self.now_angvel = (self.now_angle - self.pre_angle) / time_interval
                self.now_angvel = self.pre_angvel * alpha + self.now_angvel *beta
                self.pre_angle = self.now_angle
                self.pre_angvel = self.now_angvel
                if self.now_angvel > self.max_angvel:
                    self.max_angvel = self.now_angvel
                # 转向半径
                # # Todo：修改计算转向半径的方法
                # if abs(self.now_angvel) < 0.1:
                #     self.turn_radius = 0
                # else:
                #     self.turn_radius = self.now_velocity_norm / self.now_angvel
                #     self.average_turn_radius = 0.9*self.average_turn_radius + 0.1*abs(self.turn_radius)
                # if self.turn_radius < self.min_turn_radius:
                #     self.min_turn_radius = self.turn_radius
                # 基于三个点计算外接圆的转向半径计算方法, 暂时计算得不是特别准确
                self.turn_radius_cnt = self.turn_radius_cnt + 1
                if self.turn_radius_cnt > 10: # 这个参数是需要好好设置的
                    self.turn_radius_cnt = 0
                    self.point_3 = self.point_2
                    self.point_2 = self.point_1
                    self.point_1 = self.now_position
                Sarea = ((self.point_2[0]-self.point_1[0])*(self.point_3[1]-self.point_1[1]) - \
                        (self.point_3[0]-self.point_1[0])*(self.point_2[1]-self.point_1[1]))/2
                AB = np.linalg.norm(self.point_2-self.point_1)
                BC = np.linalg.norm(self.point_3-self.point_2)
                AC = np.linalg.norm(self.point_3-self.point_1)
                if AB*BC*AC == 0:
                    self.turn_radius = 1000
                elif Sarea < 0.0001:
                    self.turn_radius = 1000
                else:
                    self.turn_radius = AB*BC*AC/Sarea
                    self.average_turn_radius = 0.9*self.average_turn_radius + 0.1*abs(self.turn_radius)
                if self.turn_radius < self.min_turn_radius:
                    self.min_turn_radius = self.turn_radius

            self.detect_success_flag = True
            # 更新KCF的跟踪框
            self.roi_reset_cnt = self.roi_reset_cnt + 1
            if self.roi_reset_cnt > self.roi_reset_num and self.tracking_mode_flag:
                # Todo:按时修改KCF的roi，KCF容易跟丢，不知道要不要加kalman滤波
                # 更新跟踪框的机制需要改进
                center_x = int((red_x + yellow_x)/2)
                center_y = int((red_y + yellow_y)/2)
                roi_width = abs(red_x - yellow_x)*3
                if roi_width < 50:
                    roi_width = 50
                roi_height = abs(red_y - yellow_y)*3
                if roi_height < 50:
                    roi_height = 50
                
                if roi_width > 200:
                    self.tracking_mode_flag = False
                    self.kcfroi = None
                elif roi_height > 200:
                    self.tracking_mode_flag = False
                    self.kcfroi = None
                else:
                    self.kcfroi = (int(center_x-roi_width/2), int(center_y-roi_height/2), roi_width, roi_height)
                    self.tracker = imageprocess.Tracker(tracker_type="KCF")
                    self.tracker.initWorking(self.image, self.kcfroi)
                    self.roi_reset_cnt = 0
        else:
            self.detect_success_flag = False
  
        # 绘图
        # 坐标系
        origin_point = (25,15)
        x_point = (75, 15)
        y_point = (25, 65)
        img = cv2.line(img, origin_point, x_point, (255, 255, 255), 2, 4)
        img = cv2.line(img, origin_point, y_point, (255, 255, 255), 2, 4)
        img = cv2.circle(img, origin_point, 2, (255,255,255), 2)
        img = cv2.circle(img, x_point, 2, (255,255,255), 2)
        img = cv2.circle(img, y_point, 2, (255,255,255), 2)
        cv2.putText(img, 'x', (85, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, 'y', (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
 
        if self.detectstamp_show_flag:
            #
            origin_point = (641+25,471+15)
            x_point = (641+75, 471+15)
            y_point = (641+25, 471+65)
            img = cv2.line(img, origin_point, x_point, (255, 0, 0), 2, 4)
            img = cv2.line(img, origin_point, y_point, (0, 255, 0), 2, 4)
            img = cv2.circle(img, origin_point, 2, (255,0,0), 2)
            img = cv2.circle(img, x_point, 2, (255,0,0), 2)
            img = cv2.circle(img, y_point, 2, (0,255,0), 2)
            cv2.putText(img, 'x', (641+85, 471+20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
            cv2.putText(img, 'y', (641+20, 471+85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # 显示红色marker
            if red_ellipse is not None:
                img = cv2.circle(img, (red_x, red_y), 5, (255,255,0), 2)
                # if red_point_pos is not None:
                #     pos_str = "pos:[" + str(int(red_point_pos[0]*1000)) + ", " + str(int(red_point_pos[1]*1000))  + "]"
                #     cv2.putText(img, pos_str, (red_x, red_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # 显示黄色marker
            if yellow_ellipse is not None: 
                img = cv2.circle(img, (yellow_x, yellow_y), 5, (255,0,0), 2)
                # if yellow_point_pos is not None:
                #     pos_str = "pos:[" + str(int(yellow_point_pos[0]*1000)) + ", " + str(int(yellow_point_pos[1]*1000)) + "]"
                #     cv2.putText(img, pos_str, (yellow_x, yellow_y + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            # 显示速度、转向半径、机器鱼朝向、机器鱼位置
            if self.detect_success_flag:
                vel_x_str = "Max velocity norm: " + ('%.2f' % self.now_velocity_norm) + " m/s"
                # radius_str = "Min turning radius: " + ('%.2f' % (abs(self.average_turn_radius)))
                cv2.putText(img, vel_x_str, (110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # cv2.putText(img, radius_str, (110, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                # 运动坐标系
                move_origin_point = (int((red_x+yellow_x)/2),int((red_y+yellow_y)/2))
                move_x_point = (red_x-move_origin_point[0],red_y-move_origin_point[1])
                move_x_point_norm = math.sqrt(move_x_point[0]*move_x_point[0] + move_x_point[1]*move_x_point[1])
                if move_x_point_norm > 0.01:
                    move_x_axis = (int(move_origin_point[0]+move_x_point[0]/move_x_point_norm*40), int(move_origin_point[1]+move_x_point[1]/move_x_point_norm*40))
                    img = cv2.line(img, move_origin_point, move_x_axis, (0, 0, 255), 2, 4)
                    img = cv2.circle(img, move_origin_point, 2, (0,0,255), 4)
                    pos_str = "pos:[" + str(int((yellow_point_pos[0]+red_point_pos[0])/2*1000)) + ", " + str(int((yellow_point_pos[1]+red_point_pos[1])/2*1000)) + "] mm"
                    cv2.putText(img, pos_str, (move_origin_point[0], move_origin_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        result = img
        return result

    def send_rflink_data(self):
        # Todo:加入要发送什么的指令和数据
        data = struct.pack("f", self.now_position[0]) + struct.pack("f", self.now_position[1]) + struct.pack("f", self.now_angle)
        datapack = self.rftool.RFLink_packdata(rflink.Command.SET_TARGET_POS.value, data)
        try:
            self.serialtool.write_cmd(datapack)
        except serial.serialutil.SerialException:
            print("Serial send error..")

    def add_time(self, img):
        time_str = "Time: " + "%.2f" % self.camera_time + " s"
        cv2.putText(img, time_str, (1085, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        return img

    def add_border(self, img):
        p1 = (self.cam_border[0], self.cam_border[2])
        p2 = (self.cam_border[1], self.cam_border[2])
        p3 = (self.cam_border[1], self.cam_border[3])
        p4 = (self.cam_border[0], self.cam_border[3])
        img = cv2.line(img, p1, p2, (0, 0, 0), 3, 4)
        img = cv2.line(img, p2, p3, (0, 0, 0), 3, 4)
        img = cv2.line(img, p3, p4, (0, 0, 0), 3, 4)
        img = cv2.line(img, p4, p1, (0, 0, 0), 3, 4)
        return img

    def add_cameraparam(self, img):
        worldpoints = np.array([[1, 1, self.target_depth], [1, -1, self.target_depth], [-1, -1, self.target_depth], [-1, 1, self.target_depth],\
                               [2, 2, self.target_depth], [2, -2, self.target_depth], [-2, -2, self.target_depth], [-2, 2, self.target_depth],\
                               [0.5, 0.5, self.target_depth], [0.5, -0.5, self.target_depth], [-0.5, -0.5, self.target_depth], [-0.5, 0.5, self.target_depth],\
                               [1.5, 1.5, self.target_depth], [1.5, -1.5, self.target_depth], [-1.5, -1.5, self.target_depth], [-1.5, 1.5, self.target_depth]])
        rvec = np.zeros((3,1))
        tvec = np.zeros((3,1))
        imagepoints, jacobian = cv2.projectPoints(worldpoints, rvec, tvec, self.cam_K, self.cam_D)
        for i in range(4):
            p1 = (int(imagepoints[i*4][0][0]), int(imagepoints[i*4][0][1]))
            p2 = (int(imagepoints[i*4+1][0][0]), int(imagepoints[i*4+1][0][1]))
            p3 = (int(imagepoints[i*4+2][0][0]), int(imagepoints[i*4+2][0][1]))
            p4 = (int(imagepoints[i*4+3][0][0]), int(imagepoints[i*4+3][0][1]))
            img = cv2.line(img, p1, p2, (0, 0, 255), 1, 4)
            img = cv2.line(img, p2, p3, (0, 0, 255), 1, 4)
            img = cv2.line(img, p3, p4, (0, 0, 255), 1, 4)
            img = cv2.line(img, p4, p1, (0, 0, 255), 1, 4)
        return img

    def open_camera_button_slot(self):
        if self.camera_flag:
            return
        # 创建一个device
        # create a device manager
        self.cam_device_manager = gx.DeviceManager()
        dev_num, dev_info_list = self.cam_device_manager.update_device_list()
        if dev_num is 0:
            self.statusBar().showMessage("相机设备数量为0，无法打开相机")
            return
        # open the first device
        self.cam = self.cam_device_manager.open_device_by_index(1)
        # exit when the camera is a mono camera
        if self.cam.PixelColorFilter.is_implemented() is False:
            self.statusBar().showMessage("本程序不支持单色相机")
            self.cam.close_device()
            return
        # set trigger mode
        if dev_info_list[0].get("device_class") == gx.GxDeviceClassList.USB2:
            # set trigger mode
            self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
        else:
            # set trigger mode and trigger source
            self.cam.TriggerMode.set(gx.GxSwitchEntry.ON)
            self.cam.TriggerSource.set(gx.GxTriggerSourceEntry.SOFTWARE)
        # set exposure
        self.cam.ExposureTime.set(self.cam_exposure)
        # set gain
        self.cam.Gain.set(10.0)
        # get param of improving image quality
        if self.cam.GammaParam.is_readable():
            gamma_value = self.cam.GammaParam.get()
            self.cam_gamma_lut = gx.Utility.get_gamma_lut(gamma_value)
        else:
            self.cam_gamma_lut = None
        if self.cam.ContrastParam.is_readable():
            contrast_value = self.cam.ContrastParam.get()
            self.cam_contrast_lut = gx.Utility.get_contrast_lut(contrast_value)
        else:
            self.cam_contrast_lut = None
        if self.cam.ColorCorrectionParam.is_readable():
            self.cam_color_correction_param = self.cam.ColorCorrectionParam.get()
        else:
            self.cam_color_correction_param = 0
        # 打开camera stream
        self.cam.stream_on()
        # 开启计时器
        self.timer.start()
        self.camera_flag = True
        self.camera_start_time = time.time()
        st = self.statusbar_text()
        self.statusBar().showMessage(st)
    
    def close_camera_button_slot(self):
        if self.camera_flag is not True:
            return
        try:
            self.timer.stop()
            self.cam.stream_off()
            self.cam.close_device()
            self.camera_flag = False
            self.detection_flag = False
            self.velocity_flag = False
            self.angulvel_flag = False
            self.parameter_flag = False
            st = self.statusbar_text()
            self.statusBar().showMessage(st)
        except AttributeError:
            pass
    
    def enable_detect_button_slot(self):
        self.detection_flag = True
        st = self.statusbar_text()
        self.statusBar().showMessage(st)
    
    def disable_detect_button_slot(self):
        self.detection_flag = False
        st = self.statusbar_text()
        self.statusBar().showMessage(st)

    def yellow_marker_checkbox_slot(self):
        if self.yellow_marker_checkbox.isChecked():
            self.red_marker_checkbox.setChecked(False)
        self.refresh_hsvparam_ui()

    def red_marker_checkbox_slot(self):
        if self.red_marker_checkbox.isChecked():
            self.yellow_marker_checkbox.setChecked(False)
        self.refresh_hsvparam_ui()

    def time_checkbox_slot(self):
        if self.time_checkbox.isChecked():
            self.time_show_flag = True
        else:
            self.time_show_flag = False

    def border_checkbox_slot(self):
        if self.border_checkbox.isChecked():
            self.border_show_flag = True
        else:
            self.border_show_flag = False
    
    def cameraparam_checkbox_slot(self):
        if self.cameraparam_checkbox.isChecked():
            self.cameraparam_show_flag = True
        else:
            self.cameraparam_show_flag = False

    def detectstamp_checkbox_slot(self):
        if self.detectstamp_checkbox.isChecked():
            self.detectstamp_show_flag = True
        else:
            self.detectstamp_show_flag = False

    def background_checkbox_slot(self):
        if self.background_checkbox.isChecked():
            self.background_sub_flag = True
        else:
            self.background_sub_flag = False

    def debug_mode_button_slot(self):
        self.debugmode_flag = not self.debugmode_flag
        st = self.statusbar_text()
        self.statusBar().showMessage(st)

    def save_parameter_button_slot(self):
        itemstr = 'Others'
        if self.yellow_marker_checkbox.isChecked():
            itemstr = 'YellowTarget'
        elif self.red_marker_checkbox.isChecked():
            itemstr = 'RedTarget'
        else:
            itemstr = 'Others'
        self.hsvparam[itemstr]['Hmin'] = self.hue_low_splider.value()
        self.hsvparam[itemstr]['Hmax'] = self.hue_high_splider.value()
        self.hsvparam[itemstr]['Smin'] = self.satur_low_splider.value()
        self.hsvparam[itemstr]['Smax'] = self.satur_high_splider.value()
        self.hsvparam[itemstr]['Vmin'] = self.value_low_splider.value()
        self.hsvparam[itemstr]['Vmax'] = self.value_high_splider.value()
        with open(self.paramfilepath, "w", encoding="utf-8") as f:
            yaml.dump(self.hsvparam, f)

    def take_photo_button_slot(self):
        if self.image is not None:
            now_time = datetime.datetime.now()
            imagename = 'image/' + str(now_time.year-2000).zfill(2) + str(now_time.month).zfill(2) + str(now_time.day).zfill(2) + \
                        str(now_time.hour).zfill(2) + str(now_time.minute).zfill(2)+ str(now_time.second).zfill(2) + '.jpg'
            bgr = cv2.cvtColor(self.image, cv2.COLOR_RGB2BGR)
            cv2.imwrite(imagename, bgr)
        st = self.statusbar_text()
        self.statusBar().showMessage(st)

    def record_video_button_slot(self):
        self.recordvideo_flag = not self.recordvideo_flag
        if self.camera_flag and self.recordvideo_flag:
            self.record_video_button.setText('停止录制')
            now_time = datetime.datetime.now()
            videoname = 'video/' + str(now_time.year-2000).zfill(2) + str(now_time.month).zfill(2) + str(now_time.day).zfill(2) + \
                        str(now_time.hour).zfill(2) + str(now_time.minute).zfill(2) + str(now_time.second).zfill(2) + '.avi'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.videowriter = cv2.VideoWriter(videoname, fourcc, self.video_frame_rate, (1292, 964))
            filename = 'video/' + str(now_time.year-2000).zfill(2) + str(now_time.month).zfill(2) + str(now_time.day).zfill(2) + \
                        str(now_time.hour).zfill(2) + str(now_time.minute).zfill(2) + str(now_time.second).zfill(2) + '.txt'
            self.filewriter = open(filename, 'w')
        else:
            self.recordvideo_flag = False
            self.record_video_button.setText('录制视频')
            self.videowriter.release()
            self.filewriter.close()
        st = self.statusbar_text()
        self.statusBar().showMessage(st)

    def save_background_button_slot(self):
        if self.image is not None:
            now_time = datetime.datetime.now()
            imagename = 'background.jpg'
            cv2.imwrite(imagename, self.image)
            self.background = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        st = self.statusbar_text()
        self.statusBar().showMessage(st)

    def set_kcfroi_button_slot(self):
        x0, y0, x1, y1 = self.image_window.getRect()
        # print(x0, y0, x1, y1)
        if x0==0 and x1==0:
            return
        if self.image is None:
            return
        self.kcfroi = (x0, y0, x1-x0, y1-y0)
        self.tracker = imageprocess.Tracker(tracker_type="KCF")
        self.tracker.initWorking(self.image, self.kcfroi)
        self.image_window.clearRect()
        self.tracking_mode_flag = True
        pass

    def serial_open_button_slot(self):
        """
        串口打开按钮对应的槽函数
        :return:
        """
        port = self.serial_com_combo.currentText()
        baud = int(self.serial_bps_combo.currentText())
        try:
            self.serialtool.init_serial(port,baud)
            self.serialport_flag = True
            st = self.statusbar_text()
            self.statusBar().showMessage(st)
        except serial.serialutil.SerialException:
            self.serialport_flag = False
            self.statusBar().showMessage('串口不存在')


    def serial_close_button_slot(self):
        """
        串口关闭对应的槽函数
        :return:
        """
        self.serialtool.close_serial()
        self.serialport_flag = False
        st = self.statusbar_text()
        self.statusBar().showMessage(st)

    def refresh_hsvparam_ui(self):
        itemstr = 'Others'
        if self.yellow_marker_checkbox.isChecked():
            itemstr = 'YellowTarget'
        elif self.red_marker_checkbox.isChecked():
            itemstr = 'RedTarget'
        else:
            itemstr = 'Others'
        self.hue_low_splider.setValue(self.hsvparam[itemstr]['Hmin'])
        self.hue_high_splider.setValue(self.hsvparam[itemstr]['Hmax'])
        self.satur_low_splider.setValue(self.hsvparam[itemstr]['Smin'])
        self.satur_high_splider.setValue(self.hsvparam[itemstr]['Smax'])
        self.value_low_splider.setValue(self.hsvparam[itemstr]['Vmin'])
        self.value_high_splider.setValue(self.hsvparam[itemstr]['Vmax'])

    def statusbar_text(self):
        statustext = ''
        if self.camera_flag:
            statustext = statustext + '相机已打开，'
            if self.detection_flag:
                pass
            else:
                statustext = statustext + '检测未启用，'
        else:
            statustext = statustext + '相机未打开, '

        if self.serialport_flag:
            statustext = statustext + '串口已启用, '
        else:
            statustext = statustext + '串口未启用, '

        if self.debugmode_flag:
            statustext = statustext + '调试模式'
        else:
            statustext = statustext + '正常模式'

        return statustext


    def closeEvent(self, event):
        self.close_signal.emit()
        self.close()

if __name__ == '__main__':

    app = QtWidgets.QApplication(sys.argv)  # 创建QApplication对象是必须，管理整个程序，参数可有可无，有的话可接收命令行参数

    globalvision = GLOBALVISION()  # 创建窗体对象
    
    sys.exit(app.exec_())
