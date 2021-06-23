import serial

class Serial():
    """
    串口通讯类
    """
    def __init__(self):
        """
        创建串口对象
        """
        self.ser = serial.Serial()

    def init_serial(self, port, baudrate):
        """
        打开串口
        :param port:设备名称
        :param baudrate:波特率
        :return:
        """
        self.ser.port = port
        self.ser.baudrate = baudrate
        self.ser.open()

    def close_serial(self):
        """
        关闭串口
        :return:
        """
        self.ser.close()

    def write_cmd(self, cmd):
        """
        串口发送指令
        :param cmd: RFLink消息
        :return:
        """
        self.ser.write(cmd)

    def read_data(self):
        """
        串口接收数据
        :return:串口接收到的数据(byte类型)
        """
        rx_data = self.ser.read()
        return rx_data


# if __name__ == "__main__":
#     ser = RobotSerial()
#     ser.init_serial('/dev/ttyUSB1',9600)
#     cmd = b"\xff\xff\x11\x01\x00\x00\x01\x13"
#     ser.write_cmd(cmd)
#     ser.read_data()