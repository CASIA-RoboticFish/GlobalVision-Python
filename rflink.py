from enum import Enum

MY_ID = b'\x11'
FRIEND_ID = b'\x01'

# Recstate枚举类型
Recstate = Enum('Recstate',(\
	'WAITING_FF',\
	'SENDER_ID',\
	'RECEIVER_ID',\
	'RECEIVE_LEN',\
	'RECEIVE_PACKAGE',\
	'RECEIVE_CHECK'))


# Command枚举类型
Command = Enum('Command',(\
    # 系统相关指令
    'SHAKING_HANDS',\
    'SYNCHRONIZE_CLOCK',\
    # 游动控制指令
    'SET_SWIM_RUN',\
    'SET_SWIM_STOP',\
    'SET_SWIM_FORCESTOP',\
    'SET_SWIM_SPEEDUP',\
    'SET_SWIM_SPEEDDOWN',\
    'SET_SWIM_LEFT',\
    'SET_SWIM_RIGHT',\
    'SET_SWIM_FORWARD',\
    'SET_SWIM_BACKWARD',\
    'SET_SWIM_UP',\
    'SET_SWIM_DOWN',\
    'SET_SWIM_ASCEND',\
    'SET_SWIM_DESCEND',\
    'SET_SWIM_LEFT_BYPF',\
    'SET_SWIM_RIGHT_BYPF',\
    'SET_SWIM_CPG_FREQ',\
	'SET_SWIM_CPG_AMP',\
	'SET_SWIM_CPG_OFFSET',\
    # 尾鳍控制指令
    'SET_TAIL_CPG_FREQ',\
    'SET_TAIL_CPG_AMP',\
    'SET_TAIL_CPG_OFFSET',\
    'SET_TAIL_RUN',\
    'SET_TAIL_STOP',\
    'SET_TAIL_SPEEDUP',\
	'SET_TAIL_SPEEDDOWN',\
    'SET_TAIL_ANGLE',\
    'SET_TAIL_ANGLE_ZERO',\
    'SET_TAIL_ANGLE_ADD',\
    'SET_TAIL_ANGLE_ABATE',\
    'SET_TAIL_PWM',\
    # 胸鳍控制指令
    'SET_PECTFIN_MODE',\
    'SET_PECTFIN_FORWARD_MODE',\
    'SET_PECTFIN_BACKWARD_MODE',\
    'SET_PECTFIN_UPWARD_MODE',\
    'SET_PECTFIN_LEFTSTEER_MODE',\
    'SET_PECTFIN_RIGHTSTEER_MODE',\
    'SET_PECTFIN_CPG_FREQ',\
    'SET_PECTFIN_CPG_AMP',\
    'SET_PECTFIN_CPG_OFFSET',\
    'SET_PECTFIN_RUN',\
    'SET_PECTFIN_STOP',\
    'SET_PECTFIN_SPEEDUP',\
    'SET_PECTFIN_SPEEDDOWN',\
    'SET_PECTFIN_ANGLE',\
    'SET_PECTFIN_ANGLE_ADD',\
	'SET_PECTFIN_ANGLE_ZERO',\
	'SET_PECTFIN_ANGLE_ABATE',\
    'SET_LEFTPECTFIN_CPG_FREQ',\
    'SET_LEFTPECTFIN_CPG_AMP',\
    'SET_LEFTPECTFIN_CPG_OFFSET',\
    'SET_LEFTPECTFIN_RUN',\
    'SET_LEFTPECTFIN_STOP',\
    'SET_LEFTPECTFIN_SPEEDUP',\
    'SET_LEFTPECTFIN_SPEEDDOWN',\
    'SET_LEFTPECTFIN_ANGLE',\
    'SET_LEFTPECTFIN_ANGLE_ADD',\
	'SET_LEFTPECTFIN_ANGLE_ZERO',\
	'SET_LEFTPECTFIN_ANGLE_ABATE',\
    'SET_LEFTPECTFIN_PWM',\
    'SET_RIGHTPECTFIN_CPG_FREQ',\
    'SET_RIGHTPECTFIN_CPG_AMP',\
    'SET_RIGHTPECTFIN_CPG_OFFSET',\
    'SET_RIGHTPECTFIN_RUN',\
    'SET_RIGHTPECTFIN_STOP',\
    'SET_RIGHTPECTFIN_SPEEDUP',\
    'SET_RIGHTPECTFIN_SPEEDDOWN',\
    'SET_RIGHTPECTFIN_ANGLE',\
    'SET_RIGHTPECTFIN_ANGLE_ADD',\
	'SET_RIGHTPECTFIN_ANGLE_ZERO',\
	'SET_RIGHTPECTFIN_ANGLE_ABATE',\
    'SET_RIGHTPECTFIN_PWM',\
    # 静态吸附指令
    'SET_STATIC_ATTACH_START',\
    'SET_STATIC_ATTACH_STOP',\
    # 动态吸附指令
    'SET_DYNAMIC_ATTACH_START',\
    'SET_DYNAMIC_ATTACH_STOP',\
    # 电磁阀控制指令
    'SET_VALVE_ON',\
    'SET_VALVE_OFF',\
    'SET_VALVE1_ON',\
    'SET_VALVE1_OFF',\
    'SET_VALVE2_ON',\
    'SET_VALVE2_OFF',\
    # 水泵控制指令
    'SET_PUMP_ON',\
    'SET_PUMP_OFF',\
    'SET_PUMP_IN_ON',\
    'SET_PUMP_IN_OFF',\
    'SET_PUMP_OUT_ON',\
    'SET_PUMP_OUT_OFF',\
    # 云台控制指令
    'SET_GIMBAL_RUN',\
    'SET_GIMBAL_STOP',\
    'SET_GIMBAL_ZERO',\
    'SET_GIMBAL_ANGLE',\
    # 飞轮控制指令
    'SET_FLYWHEEL_RUN',\
    'SET_FLYWHEEL_STOP',\
    'SET_FLYWHEEL_DATA',\
    'SET_FLYWHEEL_CMD',\
    # 吸排水控制指令
    'SET_BALLAST_INIT',\
    'SET_BALLAST_POS',\
    'SET_BALLAST_POS_ADD',\
    'SET_BALLAST_POS_ZERO',\
    'SET_BALLAST_POS_ABATE',\
    'SET_LEFTBALLAST_POS',\
    'SET_LEFTBALLAST_PWM',\
    'SET_RIGHTBALLAST_POS',\
    'SET_RIGHTBALLAST_PWM',\
    # 相机控制指令
    'SET_CAMERA_ANGLE',\
    'SET_CAMERA_PWM',\
    # 其他控制指令
    'SET_TARGET_POS',\
    'SET_GLOBAL_POS',\
    'SET_DATASHOW_OVER',\
    'SET_APP_START',\
    'SET_APP_STOP',\
    'SET_DEPTH_TARGET',\
    'SET_DEPTHCTL_K1',\
    'SET_DEPTHCTL_K2',\
    'SET_DEPTHPID_P',\
    'SET_DEPTHPID_I',\
    'SET_DEPTHPID_D',\
    'SET_PID_P',\
    'SET_PID_I',\
    'SET_PID_D',\
    'SET_CTL_PARAM',\
    # 读取状态指令
    'READ_ROBOT_STATUS',\
    'READ_MOTION_PARAM',\
    'READ_CPG_PARAM',\
    'READ_SINE_MOTION_PARAM',\
    'READ_IMU1_ATTITUDE',\
    'READ_IMU1_ACCEL',\
    'READ_IMU1_GYRO',\
    'READ_IMU2_ATTITUDE',\
    'READ_IMU2_ACCEL',\
    'READ_IMU2_GYRO',\
    'READ_VARISTOR1_VAL',\
    'READ_VARISTOR2_VAL',\
    'READ_GIMBAL1_ANGLE',\
    'READ_GIMBAL2_ANGLE',\
    'READ_FLYWHEEL_ANGLE',\
    'READ_FLYWHEEL_VEL',\
    'READ_DEPTH',\
    'READ_GENERAL_DATA',\
    'READ_FILE_LIST',\
    'GOTO_ATTACH',\
    'GOTO_DETACH',\
    'GOTO_STORAGE_DATA',\
    'GOTO_STOP_STORAGE',\
    'GOTO_SEND_DATA',\
    'PRINT_SYS_MSG',\
    'LAST_COMMAND_FLAG'))


class RFLink():
    """
    Robotic Fish 通讯协议类
    通讯协议规范:(一帧完整数据如下:)
    0xFF, 0xFF, SENDER_ID, RECEIVER_ID, MESSAGE_LEN_H, MESSAGE_LEN_L, COMMAND, MESSAGE, CHECKNUM

    :arg sender_id: 发送者ID
    :arg receiver_id: 接收者ID
    :arg length: 消息长度
    :arg message: 消息(byte类型)

    :attributes RFLink_receivedata:接收状态机,解码RFLink通讯协议
    :attributes RFLink_packdata:将待发送数据按RFLink通讯协议打包
    """
    def __init__(self):
        self.sender_id = b''
        self.receiver_id = b''
        self.length = 0
        self.message = b''
        self._receive_state = Recstate.WAITING_FF
        self._checksum = 0
        self._byte_count = 0
        self.MY_ID = MY_ID
        self.FRIEND_ID = FRIEND_ID
        


    def RFLink_receivedata(self, rx_data):
        """
        RFLink接收状态机
        :param rx_data: 串口接收到的数据
        :return: 当接收到一帧完整数据时,返回1;否则,返回0.
        """

        if self._receive_state==Recstate.WAITING_FF:
            if rx_data==b'\xff':
                self._receive_state = Recstate.SENDER_ID
                self._checksum = ord(rx_data)
                self.message = b''
                self.length = 0
                self._byte_count = 0

        elif self._receive_state==Recstate.SENDER_ID:
            if rx_data == self.FRIEND_ID:
                self._receive_state = Recstate.RECEIVER_ID
                self._checksum += ord(rx_data)
            else:
                self._receive_state = Recstate.WAITING_FF

        elif self._receive_state==Recstate.RECEIVER_ID:
            if rx_data == self.MY_ID:
                self._receive_state = Recstate.RECEIVE_LEN
                self._checksum += ord(rx_data)
            else:
                self._receive_state = Recstate.WAITING_FF

        elif self._receive_state == Recstate.RECEIVE_LEN:
            self._receive_state = Recstate.RECEIVE_PACKAGE
            self._checksum += ord(rx_data)
            self.length = ord(rx_data)

        elif self._receive_state == Recstate.RECEIVE_PACKAGE:
            self._checksum += ord(rx_data)
            self.message = self.message + rx_data
            self._byte_count += 1
            if self._byte_count > self.length:
                self._receive_state = Recstate.RECEIVE_CHECK
                self._checksum  = self._checksum % 255

        elif self._receive_state == Recstate.RECEIVE_CHECK:
            if rx_data == self._checksum.to_bytes(1,'big'):
                self._checksum = 0
                self._receive_state = Recstate.WAITING_FF
                return 1
            else:
                self._receive_state = Recstate.WAITING_FF


        else:
            self._receive_state = Recstate.WAITING_FF

        return 0



    def RFLink_packdata(self, cmd, databyte):
        """
        RFLink数据与指令打包函数
        :param cmd:Command
        :param data:待发送数据
        :return:符合RFLink通讯协议的消息包
        """
        first_byte = b'\xff'
        second_byte = self.MY_ID
        third_byte = self.FRIEND_ID

        cmdbyte = cmd.to_bytes(1,'big')
        if databyte != 0 and databyte is not None:
            datalenbyte = len(databyte).to_bytes(1,'big')
        else:
            databyte = b''
            datalenbyte = b'\x00'

        check_num = ord(first_byte) + ord(second_byte) + ord(third_byte)
        check_num = check_num + datalenbyte[0] + ord(cmdbyte)
        for data in databyte:
            check_num = check_num + data
        check_num = (check_num%255).to_bytes(1,'big')

        datapack = first_byte + second_byte + third_byte + datalenbyte + cmdbyte + databyte + check_num

        return datapack





if __name__ == "__main__":
    #print(RFLink_packdata(Command.SET_CPG_AMP.value,0.0))
    # rf = RFLink()
    # rf.RFLink_receivedata(b'\xff')
    # rf.RFLink_receivedata(b'\xff')
    # rf.RFLink_receivedata(b'\x01')
    # rf.RFLink_receivedata(b'\x11')
    # rf.RFLink_receivedata(b'\x00')
    # rf.RFLink_receivedata(b'\x00')
    # rf.RFLink_receivedata(b'\x01')
    # print(rf.RFLink_receivedata(b'\x13'))
    print(Command(132))





























