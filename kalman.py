from typing_extensions import IntVar
import numpy as np
class KalmanFilter:
    def __init__(self):
        self.A = None # 方阵，状态矩阵
        self.H = None # 量测矩阵

        self.X = None # 列向量
        self.P = None # 方阵
        self.K = None # 矩阵

        self.Q = None # 方阵 状态噪声阵
        self.R = None # 方阵 量测噪声阵
        
    def init_variable(self):
        pass

    def time_update(self):
        self.X = self.A*self.X
        self.P = self.A*self.P*self.A.T + self.Q

    def meas_update(self, Z):
        self.K = self.P*self.H.T*inv(self.H*self.P*self.H.T+R)
        