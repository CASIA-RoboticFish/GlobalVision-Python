import cv2
import numpy as np
import imageprocess


if __name__ == '__main__':
    img = cv2.imread('image/210127223747.jpg',cv2.IMREAD_COLOR)

    sum_t1 = 0
    sum_t2 = 0
    sum_t3 = 0
    sum_t4 = 0
    sum_t5 = 0
    sum_t6 = 0
    
    for i in range(50):
        t0 = cv2.getTickCount()
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) # 1ms

        t1 = cv2.getTickCount()
        sum_t1 = sum_t1 + (t1-t0)/cv2.getTickFrequency()
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) # 1ms
        blur = cv2.GaussianBlur(img, (7, 7), 0) # 4ms
        
        
        hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV) # 5ms

        print(hsv[10:950, 10:950,:])
        a = hsv[10:950, 10:950,:]
        
        color_low = np.array([40, 0, 0])
        color_high = np.array([104, 98, 133])
        border = np.array([20,920,50,900])
        thres = cv2.inRange(hsv, color_low, color_high) # 2ms

        
        

        median_filter = cv2.medianBlur(thres, 7)  # 中值滤波 28ms

        t2 = cv2.getTickCount()
        sum_t2 = sum_t2 + (t2-t1)/cv2.getTickFrequency()
        

        kernal = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        mask = cv2.morphologyEx(median_filter, cv2.MORPH_CLOSE, kernal)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernal) # 5ms

        t3 = cv2.getTickCount()
        sum_t3 = sum_t3 + (t3-t2)/cv2.getTickFrequency()
        

        mask_border = np.zeros(mask.shape, dtype='uint8')
        mask_border[border[2]:border[3],border[0]:border[1]] = mask[border[2]:border[3],border[0]:border[1]] # 0.5ms

        t4 = cv2.getTickCount()
        sum_t4 = sum_t4 + (t4-t3)/cv2.getTickFrequency()
        results = cv2.findContours(mask_border, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) # 1ms

        t5 = cv2.getTickCount()
        sum_t5 = sum_t5 + (t5-t4)/cv2.getTickFrequency()
        contours = results[1]
        
        ellipse_in_image = None
        for cnt in contours:
            if cnt.size < 25: # 轮廓中点的数量
                print('cnt size')
                print(cnt.size)
                continue
            cnt_area = cv2.contourArea(cnt)
            if cnt_area < 100: # 轮廓围成的点的面积
                print('cnt area')
                print(cnt_area)
                continue
            ellipse = cv2.fitEllipse(cnt)
            ellipse_size =  ellipse[1]
            if ellipse_size[1]/ellipse_size[0] < 0.8 or ellipse_size[1]/ellipse_size[0] > 1.2:
                print(ellipse_size[1]/ellipse_size[0])
                continue
        t6 = cv2.getTickCount()
        sum_t6 = sum_t6 + (t6-t5)/cv2.getTickFrequency()

    print(sum_t1/50*1000)
    print(sum_t2/50*1000)
    print(sum_t3/50*1000)
    print(sum_t4/50*1000)
    print(sum_t5/50*1000)
    print(sum_t6/50*1000)