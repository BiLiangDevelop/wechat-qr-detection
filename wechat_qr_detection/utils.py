# coding:utf-8
__author__ = 'rk.feng'
import math
from itertools import combinations

import cv2
import numpy as np


class CircleData(object):
    def __init__(self, center_x: int, center_y: int, radius: int):
        self.radius = radius  # 半径
        self.center_y = center_y  # 圆心 y 坐标
        self.center_x = center_x  # 圆心 x 坐标

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Circle ({}, {}) r={}>".format(self.center_x, self.center_y, self.radius)


class WechatQRInfo(object):
    def __init__(self, center_circle: CircleData, logo_circle: CircleData, c_logo: CircleData,
                 c2: CircleData, c1: CircleData):
        self.logo_circle = logo_circle
        self.center_circle = center_circle
        self.c_logo = c_logo
        self.c2 = c2
        self.c1 = c1


class WechatQRDetector(object):
    Radius_Radio_List = [1, 1.5, 3.4, 5]

    def __init__(self, img: np.ndarray):
        self.img = img

        # 圆半径
        width, height, _ = img.shape
        max_size = min(width, height)
        self.min_radius = 5  # 圆的最小半径
        self.max_radius = int(0.25 * max_size)  # 圆的最大半径
        self.min_dist = min(50, 14 * self.min_radius)  # 圆心最小间距

        # 灰度图
        self.gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # _, self.gray = cv2.threshold(cv2.blur(gray, (5,5)), 127, 255, cv2.THRESH_BINARY)

    def find_circle(self, ) -> [CircleData]:
        """ 找牛眼, 粗略
        :rtype: list of CircleData
        """
        # 霍夫变换圆检测
        raw_circles = cv2.HoughCircles(
            self.gray, cv2.HOUGH_GRADIENT, 1, self.min_dist, param1=80, param2=30,
            minRadius=self.min_radius,
            maxRadius=self.max_radius
        )

        # 模型变换
        circle_list = []
        if len(raw_circles) > 0:
            for circle in raw_circles[0]:
                circle_list.append(CircleData(center_x=int(circle[0]), center_y=int(circle[1]), radius=int(circle[2])))

        return circle_list

    def detect_qr_code(self, circle_list: [CircleData]) -> [WechatQRInfo]:
        """寻找所有的小程序二维码位置
        :rtype: list of WechatQRInfo
        """

        def check_distance(dis_pow: int, t1: CircleData, t2: CircleData):
            """ 距离与半径的关系 """
            return 7 < math.sqrt(dis_pow) / (t1.radius + t2.radius) < 70

        def find_similar_circle(target_circle: CircleData, ) -> CircleData:
            """ 寻找最合适的圆 """
            _c_list = []

            # 寻找吻合的圆
            for circle in circle_list:
                _dist = math.sqrt(
                    (circle.center_x - target_circle.center_x) ** 2 + (circle.center_y - target_circle.center_y) ** 2)
                if _dist < target_circle.radius:
                    _c_list.append((_dist, circle))

            if not _c_list:
                return

            # 最相似
            _sorted_c_list = sorted(_c_list, key=lambda x: x[0], )
            return _sorted_c_list[0][1]

        def maybe_detection_circle(t1: CircleData, t2: CircleData, t3: CircleData) -> WechatQRInfo:
            """ 是否可能是牛眼 """

            # 等腰直角三角形
            d_12_pow = (t1.center_x - t2.center_x) ** 2 + (t1.center_y - t2.center_y) ** 2
            d_13_pow = (t1.center_x - t3.center_x) ** 2 + (t1.center_y - t3.center_y) ** 2
            d_23_pow = (t3.center_x - t2.center_x) ** 2 + (t3.center_y - t2.center_y) ** 2

            dist_list = [d_12_pow, d_13_pow, d_23_pow]
            dist_list.sort()

            if abs(dist_list[0] - dist_list[1]) / dist_list[0] > 0.1 or abs(
                    dist_list[0] + dist_list[1] - dist_list[2]) / dist_list[0] > 0.1:
                return

                # 距离判断
            if not check_distance(d_12_pow, t1, t2) or not check_distance(d_13_pow, t1, t3) or \
                    not check_distance(d_23_pow, t2, t3):
                return

                # 获取直角圆
            if d_12_pow == dist_list[-1]:
                c_logo, c1, c2 = t3, t1, t2
            elif d_23_pow == dist_list[-1]:
                c_logo, c1, c2 = t1, t2, t3
            else:
                c_logo, c1, c2 = t2, t1, t3

            # 判断是否存在 logo 的圆
            _logo_target_circle = CircleData(
                center_x=int(c2.center_x + c1.center_x - c_logo.center_x),
                center_y=int(c2.center_y + c1.center_y - c_logo.center_y),
                radius=int(3 * max([c_logo.radius, c1.radius, c2.radius]))
            )
            logo_circle = find_similar_circle(_logo_target_circle)
            if logo_circle is None:
                return

            # 圆心圆

            center_circle = CircleData(
                center_x=int((c2.center_x + c1.center_x) / 2),
                center_y=int((c2.center_y + c1.center_y) / 2),
                radius=int(math.sqrt(dist_list[-1]) / 2 + logo_circle.radius)
            )
            # 半径修正
            center_circle.radius = int(math.sqrt((logo_circle.center_x - center_circle.center_x) ** 2 + (
                    logo_circle.center_y - center_circle.center_y) ** 2) + logo_circle.radius)

            # 返回候选的小程序码
            return WechatQRInfo(center_circle=center_circle, logo_circle=logo_circle, c_logo=c_logo, c1=c1, c2=c2)

        if len(circle_list) < 3:
            return []

        # 返回所有三角形组合
        triangle_list = list(combinations(circle_list, 3))

        # 根据是否是等腰直角三角形, 返回潜在的牛眼
        wechat_info_list = []
        for _circle_list in triangle_list:
            wechat_info = maybe_detection_circle(_circle_list[0], _circle_list[1], _circle_list[2])
            if wechat_info:
                wechat_info_list.append(wechat_info)

        return wechat_info_list
