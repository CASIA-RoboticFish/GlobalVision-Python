from os import error
from re import X
import cv2
import numpy as np
import math

if __name__ == '__main__':
    videoname = 'G:/机器鱼/静态吸附/位置控制实验/选择的实验/210609201201_new.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(videoname, fourcc, 25, (1292, 964))

    data = np.loadtxt('G:/机器鱼/静态吸附/位置控制实验/选择的实验/210609201201.txt')
    cap = cv2.VideoCapture("G:/机器鱼/静态吸附/位置控制实验/选择的实验/210609201201.avi")
    row_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        try:
            origin_point = (25,15)
            x_point = (75, 15)
            y_point = (25, 65)
            img = cv2.line(frame, origin_point, x_point, (255, 255, 255), 2, 4)
            img = cv2.line(frame, origin_point, y_point, (255, 255, 255), 2, 4)
            img = cv2.circle(frame, origin_point, 2, (255,255,255), 2)
            img = cv2.circle(frame, x_point, 2, (255,255,255), 2)
            img = cv2.circle(frame, y_point, 2, (255,255,255), 2)
            cv2.putText(frame, 'x', (85, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, 'y', (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)


            # 固定坐标系
            origin_point = (641+25,471+15)
            x_point = (641+75, 471+15)
            y_point = (641+25, 471+65)
            frame = cv2.line(frame, origin_point, x_point, (0, 0, 255), 2, 4)
            frame = cv2.line(frame, origin_point, y_point, (0, 255, 0), 2, 4)
            frame = cv2.circle(frame, origin_point, 2, (0,0,255), 2)
            frame = cv2.circle(frame, x_point, 2, (0,0,255), 2)
            frame = cv2.circle(frame, y_point, 2, (0,255,0), 2)
            cv2.putText(frame, 'x', (641+85, 471+20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.putText(frame, 'y', (641+20, 471+85), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)


            # 显示红色marker
            frame = cv2.circle(frame, (int(data[row_cnt, 8]), int(data[row_cnt, 9])), 5, (0,255,255), 2)
                
            # 显示黄色marker
            frame = cv2.circle(frame, (int(data[row_cnt, 10]), int(data[row_cnt, 11])), 5, (0,0,255), 2)

            # 显示速度
            now_velocity_norm = data[row_cnt, 3]*data[row_cnt, 3] + data[row_cnt, 4]*data[row_cnt, 4]
            now_velocity_norm = math.sqrt(now_velocity_norm)
            vel_x_str = "Max velocity norm: " + ('%.2f' % now_velocity_norm) + " m/s"
            cv2.putText(frame, vel_x_str, (110, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # 运动坐标系
            x = (data[row_cnt, 8] + data[row_cnt, 10])/2
            y = (data[row_cnt, 9] + data[row_cnt, 11])/2
            move_origin_point = (int(x),int(y))
            move_x_point = (data[row_cnt, 8]-move_origin_point[0],data[row_cnt, 9]-move_origin_point[1])
            move_x_point_norm = math.sqrt(move_x_point[0]*move_x_point[0] + move_x_point[1]*move_x_point[1])
            if move_x_point_norm > 0.01:
                move_x_axis = (int(move_origin_point[0]+move_x_point[0]/move_x_point_norm*60), int(move_origin_point[1]+move_x_point[1]/move_x_point_norm*60))
                frame = cv2.line(frame, move_origin_point, move_x_axis, (255, 0, 0), 2, 4)
                frame = cv2.circle(frame, move_origin_point, 2, (255,0,0), 4)
                pos_str = "pos:[" + str(int(data[row_cnt, 5]*1000)) + ", " + str(int(data[row_cnt, 6]*1000)) + "] mm"
                cv2.putText(img, pos_str, (move_origin_point[0], move_origin_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
            row_cnt = row_cnt + 1

            videowriter.write(frame)
            cv2.imshow('frame',frame)
            if cv2.waitKey(25) & 0xFF == ord('q'):
                break
        except:
            print('error')
            break

    cap.release()
    cv2.destroyAllWindows()