import math
import os
import random

import numpy as np


def logging(msg, fp, print_msg=False):
    if print_msg:
        print(msg)
    mkdir(os.path.dirname(fp))

    with open(fp, 'a') as f:
        f.write(msg + "\n")


def mkdir(fp):
    if not os.path.exists(fp):
        os.makedirs(fp)


def generate_random(num):
    alphabet = 'abcdefghijklmnopqrstuvwxyz_-ABCDEFGHIJKLMNOPQRSTUVWXYZ'
    return ''.join([random.choice(alphabet) for i in range(num)])


def point_distance_line(p, lp1, lp2):
    """
    计算点到线的垂直距离
    Args:
        p: [float, float]
        lp1: [float, float]
        lp2: [float, float]

    Returns: int

    """
    point = np.array(p)
    line_point1 = np.array(lp1)
    line_point2 = np.array(lp2)
    # 计算向量
    vec1 = line_point1 - point
    vec2 = line_point2 - point
    distance = np.abs(np.cross(vec1, vec2)) / np.linalg.norm(line_point1 - line_point2)
    return distance


def point_point_distance(p1, p2):
    """
        计算两点距离
        Args:
            p1: [float, float]
            p2: [float, float]
        Returns: float
    """
    return math.hypot(p2[0] - p1[0], p2[1] - p1[1])


def get_parallel_line(p, line):
    """
        计算平行线
        Args:
            p: [float, float]
            line: [x1, y1, x2, y2]
        Returns: point [0, b]
    """
    k = (line[3] - line[1]) / (line[2] - line[0])
    b = p[1] - k*p[0]
    return [0, b]


def cross_point(line1, line2):
    """
    计算交点
    Args:
        line1: [x1,y1,x2,y2]
        line2: [x1,y1,x2,y2]

    Returns: [x, y]
    """
    x1 = line1[0]  # 取四点坐标
    y1 = line1[1]
    x2 = line1[2]
    y2 = line1[3]

    x3 = line2[0]
    y3 = line2[1]
    x4 = line2[2]
    y4 = line2[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)  # 计算k1,由于点均为整数，需要进行浮点数转化
    b1 = y1 * 1.0 - x1 * k1 * 1.0  # 整型转浮点型是关键
    if (x4 - x3) == 0:  # L2直线斜率不存在操作
        k2 = None
        b2 = 0
    else:
        k2 = (y4 - y3) * 1.0 / (x4 - x3)  # 斜率存在操作
        b2 = y3 * 1.0 - x3 * k2 * 1.0

    if k2 is None:
        x = x3
    else:
        x = (b2 - b1) * 1.0 / (k1 - k2)
    y = k1 * x * 1.0 + b1 * 1.0
    return [x, y]


def get_vertical_line(line, ratio):
    """
    获取线段某个比例位置的垂线
    Args:
        line: [x1,y1,x2,y2]
        ratio: 所求点对于线段长度的比例

    Returns: [x1,y1,x2,y2]

    """
    x1 = line[0]  # 取四点坐标
    y1 = line[1]
    x2 = line[2]
    y2 = line[3]

    k1 = (y2 - y1) * 1.0 / (x2 - x1)
    k2 = -1 / k1
    b2 = y1 - k2 * x1

    point1_x = (x2 - x1) * ratio + x1
    point1_y = (y2 - y1) * ratio + y1
    point2_x = 0
    point2_y = b2

    return [point1_x, point1_y, point2_x, point2_y]

# "test"
# print(point_point_distance([2,-3], [3,1]))
