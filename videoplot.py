from re import X
import cv2
import numpy as np
import math

if __name__ == '__main__':
    videoname = 'video/210611212940_new.avi'
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    videowriter = cv2.VideoWriter(videoname, fourcc, 25, (1292, 964))

    data = np.loadtxt('video/210611212940.txt')
    cap = cv2.VideoCapture("video/210611212940.avi")
    row_cnt = 0
    while(cap.isOpened()):
        ret, frame = cap.read()
        
        # 固定坐标系
        origin_point = (641+25,471+15)
        x_point = (641+105, 471+15)
        y_point = (641+25, 471+95)
        frame = cv2.line(frame, origin_point, x_point, (0, 0, 255), 2, 4)
        frame = cv2.line(frame, origin_point, y_point, (0, 255, 0), 2, 4)
        frame = cv2.circle(frame, origin_point, 3, (0,0,0), 2)
        frame = cv2.circle(frame, x_point, 3, (0,0,255), 2)
        frame = cv2.circle(frame, y_point, 3, (0,255,0), 2)
        cv2.putText(frame, 'x', (641+120, 471+20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
        cv2.putText(frame, 'y', (641+20, 471+130), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 运动坐标系
        x = (data[row_cnt, 8] + data[row_cnt, 10])/2
        y = (data[row_cnt, 9] + data[row_cnt, 11])/2
        move_origin_point = (int(x),int(y))
        move_x_point = (data[row_cnt, 8]-move_origin_point[0],data[row_cnt, 9]-move_origin_point[1])
        move_x_point_norm = math.sqrt(move_x_point[0]*move_x_point[0] + move_x_point[1]*move_x_point[1])
        if move_x_point_norm > 0.01:
            move_x_axis = (int(move_origin_point[0]+move_x_point[0]/move_x_point_norm*60), int(move_origin_point[1]+move_x_point[1]/move_x_point_norm*60))
            frame = cv2.line(frame, move_origin_point, move_x_axis, (255, 0, 0), 2, 4)
            frame = cv2.circle(frame, move_origin_point, 3, (255,0,0), 4)
            # pos_str = "pos:[" + str(int((yellow_point_pos[0]+red_point_pos[0])/2*1000)) + ", " + str(int((yellow_point_pos[1]+red_point_pos[1])/2*1000)) + "] mm"
            # cv2.putText(img, pos_str, (move_origin_point[0], move_origin_point[1] + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
        row_cnt = row_cnt + 1

        videowriter.write(frame)
        cv2.imshow('frame',frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()