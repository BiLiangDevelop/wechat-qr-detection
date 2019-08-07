# coding:utf-8
__author__ = 'rk.feng'
import logging
import math
import os
import re
from enum import Enum
from itertools import combinations

import cv2
import numpy as np
import pytesseract


class CircleData(object):
    def __init__(self, center_x: int, center_y: int, radius: int):
        self.radius = radius  # 半径
        self.center_y = center_y  # 圆心 y 坐标
        self.center_x = center_x  # 圆心 x 坐标

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Circle ({}, {}) r={}>".format(self.center_x, self.center_y, self.radius)

    @classmethod
    def distance(cls, t1, t2) -> float:
        """ 计算两个圆的距离 """
        return math.sqrt((t1.center_x - t2.center_x) ** 2 + (t1.center_y - t2.center_y) ** 2)


class WechatQRInfo(object):
    def __init__(self, center_circle: CircleData, logo_circle: CircleData, c_logo: CircleData,
                 c2: CircleData, c1: CircleData):
        self.logo_circle = logo_circle
        self.center_circle = center_circle
        self.c_logo = c_logo
        self.c2 = c2
        self.c1 = c1


class LogoMethod(Enum):
    CHAR_TESSERACT = 0
    IMAGE_MATCH = 1
    CONTOURS_MATCH = 2


class WechatQRDetector(object):
    Radius_Radio_List = [1, 1.5, 3.4, 5]

    def __init__(self, img: np.ndarray, logger: logging.Logger = None):
        self.logger = logger or logging.getLogger(self.__class__.__name__)
        self.img = img

        # 圆半径
        width, height, _ = img.shape
        max_size = min(width, height)
        self.min_radius = 5  # 圆的最小半径
        self.max_radius = int(0.25 * max_size)  # 圆的最大半径
        self.min_dist = min(50, 14 * self.min_radius)  # 圆心最小间距

        # 灰度图
        self.gray = WechatQRDetector.image_to_gray(image=self.img)

        # logo template
        self._log_template = None

    @classmethod
    def image_to_gray(cls, image: np.ndarray) -> np.ndarray:
        """ 图像灰度化 """
        # gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # gray = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        # _, self.gray = cv2.threshold(cv2.blur(gray, (5,5)), 127, 255, cv2.THRESH_BINARY)
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @classmethod
    def image_to_contour(cls, image: np.ndarray, is_gray: bool = False, thickness: int = 1) -> np.ndarray:
        """ 图像转轮廓图 """
        if is_gray:
            gray_image = image
        else:
            gray_image = WechatQRDetector.image_to_gray(image=image)
        ret, binary = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)  # 转换为二值图像
        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 提取轮廓

        # contour_image
        contour_image = np.zeros_like(gray_image, dtype=gray_image.dtype)
        contour_image.fill(255)
        cv2.drawContours(image=contour_image, contours=contours, contourIdx=-1, color=(0, 0, 0), thickness=thickness)
        return contour_image

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
        if raw_circles is not None and len(raw_circles) > 0:
            for circle in raw_circles[0]:
                circle_list.append(CircleData(center_x=int(circle[0]), center_y=int(circle[1]), radius=int(circle[2])))

        return circle_list

    def detect_qr_code(self, circle_list: [CircleData]) -> [WechatQRInfo]:
        """寻找所有的小程序二维码位置
        :rtype: list of WechatQRInfo
        """
        def is_tri_90(t1: CircleData, t2: CircleData, t3: CircleData) -> bool:
            """ 是否是直角三角形 """
            # 等腰直角三角形, t3 为直角点
            d_12_pow = (t1.center_x - t2.center_x) ** 2 + (t1.center_y - t2.center_y) ** 2
            d_13_pow = (t1.center_x - t3.center_x) ** 2 + (t1.center_y - t3.center_y) ** 2
            d_23_pow = (t3.center_x - t2.center_x) ** 2 + (t3.center_y - t2.center_y) ** 2

            if abs(d_13_pow - d_23_pow) / d_13_pow > 0.1 or abs(d_13_pow - d_23_pow - d_12_pow) / d_13_pow > 0.1:
                return False

            return True

        def check_distance(dis_pow: int, t1: CircleData, t2: CircleData):
            """ 距离与半径的关系 """
            return 6 < math.sqrt(dis_pow) / (t1.radius + t2.radius) < 70  # todo 有问题

        def find_similar_circle(target_circle: CircleData, ) -> CircleData:
            """ 寻找最合适的圆 """
            _c_list = []

            # 寻找吻合的圆
            for circle in circle_list:
                _dist = math.sqrt(
                    (circle.center_x - target_circle.center_x) ** 2 + (circle.center_y - target_circle.center_y) ** 2)
                if _dist < target_circle.radius:
                    _c_list.append((logo_circle_dict[circle], circle))

            if not _c_list:
                return

            # 最相似
            _sorted_c_list = sorted(_c_list, key=lambda x: x[0], reverse=True)
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

        def maybe_qrcode(left_circle: CircleData, up_circle: CircleData, up_left_circle: CircleData,
                         center_circle: CircleData, logo_circle: CircleData) -> WechatQRInfo:
            # 三个牛眼存在的情况先排除
            if left_circle and up_circle and up_left_circle:
                return

            # 必须存在至少两个牛眼
            _valid_count = len([_circle for _circle in [[left_circle, up_circle, up_left_circle]] if _circle])
            if _valid_count < 2:
                return

            # 情形 1: left + up_left
            if left_circle and up_left_circle:
                # 不可能是直角三角形
                if is_tri_90(up_left_circle, logo_circle, left_circle) is False:
                    return

                    # 两个牛眼的距离是距离比例合理
                dis_l_ul = math.sqrt(
                    (left_circle.center_x - up_left_circle.center_x) ** 2 +
                    (left_circle.center_y - up_left_circle.center_y) ** 2
                )
                if dis_l_ul / (left_circle.radius + up_left_circle.radius) < 6 or \
                        dis_l_ul / (left_circle.radius + up_left_circle.radius) > 70:
                    return

                    # 如果存在 center_circle
                if center_circle:
                    # todo 圆心圆的使用
                    pass

                # 构造二维码位置
                return WechatQRInfo(
                    center_circle=CircleData(
                        center_x=int((logo_circle.center_x + up_left_circle.center_x) / 2),
                        center_y=int((logo_circle.center_y + up_left_circle.center_y) / 2),
                        radius=int(math.sqrt((logo_circle.center_y - up_left_circle.center_y) ** 2 + (
                                logo_circle.center_x - up_left_circle.center_x) ** 2) / 2 + logo_circle.radius)
                    ),
                    logo_circle=logo_circle,
                    c_logo=up_left_circle,
                    c1=left_circle,
                    c2=CircleData(
                        center_x=logo_circle.center_x,
                        center_y=up_left_circle.center_y,
                        radius=int((left_circle.radius + up_left_circle.radius) / 2)
                    )
                )

            # 情形 2: left + up
            if left_circle and up_circle:
                # 不可能是直角三角形
                if is_tri_90(left_circle, up_circle, logo_circle) is False:
                    return

                # 两个牛眼的距离是距离比例合理
                dis_l_u = math.sqrt(
                    (left_circle.center_x - up_circle.center_x) ** 2 +
                    (left_circle.center_y - up_circle.center_y) ** 2
                )
                if dis_l_u / (left_circle.radius + up_circle.radius) < 8 or \
                        dis_l_u / (left_circle.radius + up_circle.radius) > 100:
                    return

                # 如果存在 center_circle
                if center_circle:
                    # todo 圆心圆的使用
                    pass

                # 构造二维码位置
                return WechatQRInfo(
                    center_circle=CircleData(
                        center_x=int((left_circle.center_x + up_circle.center_x) / 2),
                        center_y=int((left_circle.center_y + up_circle.center_y) / 2),
                        radius=int(dis_l_u / 2 + logo_circle.radius)
                    ),
                    logo_circle=logo_circle,
                    c_logo=CircleData(
                        center_x=left_circle.center_x,
                        center_y=up_circle.center_y,
                        radius=int((left_circle.radius + up_circle.radius) / 2)
                    ),
                    c1=left_circle,
                    c2=up_circle,
                )

            # 情形 3: up_left + up
            if up_left_circle and up_circle:
                # 不可能是直角三角形
                if is_tri_90(up_left_circle, logo_circle, up_circle) is False:
                    return

                # 两个牛眼的距离是距离比例合理
                dis_u_ul = math.sqrt(
                    (up_left_circle.center_x - up_circle.center_x) ** 2 +
                    (up_left_circle.center_y - up_circle.center_y) ** 2
                )
                if dis_u_ul / (up_left_circle.radius + up_circle.radius) < 6 or \
                        dis_u_ul / (up_left_circle.radius + up_circle.radius) > 70:
                    return

                # 如果存在 center_circle
                if center_circle:
                    # todo 圆心圆的使用
                    pass

                # 构造二维码位置
                return WechatQRInfo(
                    center_circle=CircleData(
                        center_x=int((logo_circle.center_x + up_left_circle.center_x) / 2),
                        center_y=int((logo_circle.center_y + up_left_circle.center_y) / 2),
                        radius=int(math.sqrt((logo_circle.center_y - up_left_circle.center_y) ** 2 + (
                                logo_circle.center_x - up_left_circle.center_x) ** 2) / 2 + logo_circle.radius)
                    ),
                    logo_circle=logo_circle,
                    c_logo=up_left_circle,
                    c1=CircleData(
                        center_x=up_left_circle.center_x,
                        center_y=logo_circle.center_y,
                        radius=int((up_left_circle.radius + up_circle.radius) / 2)
                    ),
                    c2=up_left_circle
                )

        def find_wechat_qrcode_by_logo_circle(logo_circle: CircleData) -> WechatQRInfo:
            """ 利用 logo circle 反查 小程序码 """
            # todo 暂时不考虑 logo 旋转的情形

            # 候选圆
            _left_circle_list = [None, ]
            _up_circle_list = [None, ]
            _up_left_circle_list = [None, ]
            _center_circle_list = [None, ]
            for _circle in circle_list:
                if _circle == logo_circle:
                    pass

                # 同一条水平线
                if abs(_circle.center_y - logo_circle.center_y) / logo_circle.radius < 0.1 and \
                        _circle.radius < logo_circle.radius and _circle.center_x < logo_circle.center_x and \
                        3 < CircleData.distance(_circle, logo_circle) / (_circle.radius + logo_circle.radius) < 7:
                    _left_circle_list.append(_circle)
                    continue

                # 同一竖线
                if abs(_circle.center_x - logo_circle.center_x) / logo_circle.radius < 0.1 and \
                        _circle.radius < logo_circle.radius and _circle.center_y < logo_circle.center_y and \
                        3 < CircleData.distance(_circle, logo_circle) / (_circle.radius + logo_circle.radius) < 7:
                    _up_circle_list.append(_circle)
                    continue

                # 可能是直角牛眼
                if _circle.radius < logo_circle.radius and \
                        _circle.center_y < logo_circle.center_y and _circle.center_x < logo_circle.center_x and \
                        0.9 < (_circle.center_x - logo_circle.center_x) / (
                        _circle.center_y - logo_circle.center_y) < 1.1 and \
                        4 < CircleData.distance(_circle, logo_circle) / (_circle.radius + logo_circle.radius) < 10:
                    _up_left_circle_list.append(_circle)
                    continue

                # 可能是程序码圆心
                if _circle.radius > logo_circle.radius and \
                        _circle.center_y < logo_circle.center_y and _circle.center_x < logo_circle.center_x and \
                        0.9 < (_circle.center_x - logo_circle.center_x) / (
                        _circle.center_y - logo_circle.center_y) < 1.1 and \
                        1 < CircleData.distance(_circle, logo_circle) / (_circle.radius + logo_circle.radius) < 7:
                    _center_circle_list.append(_circle)
                    continue

            # 搭配
            for _left_circle in _left_circle_list:
                for _up_circle in _up_circle_list:
                    for _up_left_circle in _up_left_circle_list:
                        for _center_circle in _center_circle_list:
                            # 计算有效值的个数
                            _valid_count = len(
                                [_circle for _circle in [[_left_circle, _up_circle, _up_left_circle, _center_circle]] if
                                 _circle])
                            if _valid_count < 2:
                                continue

                            # 判断组合是否有效
                            _wechat_info = maybe_qrcode(_left_circle, _up_circle, _up_left_circle, _center_circle,
                                                        logo_circle)
                            if _wechat_info:
                                return _wechat_info

        if len(circle_list) < 3:
            return []

        ###############################################
        # 1. 存在三个牛眼
        ###############################################
        # 返回所有三角形组合
        triangle_list = list(combinations(circle_list, 3))

        # 记录所有可能是logo 的圆的信息
        prob_list = self.is_logo(gray_image=self.gray, circle_list=circle_list, method=LogoMethod.IMAGE_MATCH)
        self.logger.info("prob list of is_logo is {}".format(prob_list))
        logo_circle_dict = {
            circle: prob_list[index] for index, circle in enumerate(circle_list)
        }

        # 根据是否是等腰直角三角形, 返回潜在的牛眼
        wechat_info_list = []
        for _circle_list in triangle_list:
            wechat_info = maybe_detection_circle(_circle_list[0], _circle_list[1], _circle_list[2])
            if wechat_info:
                wechat_info_list.append(wechat_info)

        # 判断
        logo_circle_list = [circle for circle, prob in logo_circle_dict.items() if prob > 0.01]
        self.logger.info("logo circle {}, qrcode {}!".format(len(logo_circle_list), len(wechat_info_list)))

        if wechat_info_list:
            return wechat_info_list

        ###############################################
        # 2. 存在 logo
        ###############################################
        # 如果存在 logo, 但没找到小程序码
        if logo_circle_list:
            # 利用 logo 寻找小程序码
            for _logo_circle in logo_circle_list:
                wechat_info = find_wechat_qrcode_by_logo_circle(logo_circle=_logo_circle)
                if wechat_info:
                    wechat_info_list.append(wechat_info)

            self.logger.info("{} qrcode found by logo circle!".format(len(wechat_info_list)))

        ###############################################
        # 3. 不存在 logo
        ###############################################
        new_logo_circle_list = self.find_logo(gray_image=self.gray, filter_by_prob=False)
        _prob_list = self.is_logo(gray_image=self.gray, circle_list=new_logo_circle_list, method=LogoMethod.IMAGE_MATCH)
        new_logo_circle_list = [circle for index, circle in enumerate(new_logo_circle_list) if _prob_list[index] > 0]
        self.logger.info("find logo by image match: {} found!".format(len(new_logo_circle_list)))
        if new_logo_circle_list:
            # 利用 logo 寻找小程序码
            for _logo_circle in new_logo_circle_list:
                wechat_info = find_wechat_qrcode_by_logo_circle(logo_circle=_logo_circle)
                if wechat_info:
                    wechat_info_list.append(wechat_info)

            self.logger.info("{} qrcode found by logo circle!".format(len(wechat_info_list)))

        return wechat_info_list

    @staticmethod
    def logo_rotate(contour_image: np.ndarray, circle: CircleData):
        """ logo 旋转 """
        tmp_radius = circle.radius * 2  # 真正的 logo circle 不会出错
        logo_image = contour_image[
                     int(circle.center_y - tmp_radius):int(circle.center_y + tmp_radius),
                     int(circle.center_x - tmp_radius):int(circle.center_x + tmp_radius)
                     ].copy()

        # 旋转
        logo_rotated_image = cv2.warpAffine(
            logo_image,
            cv2.getRotationMatrix2D((tmp_radius, tmp_radius), 45, 1.0),
            (tmp_radius * 2, tmp_radius * 2)
        )
        logo_rotated_image = logo_rotated_image[
                             int(tmp_radius - circle.radius):int(tmp_radius + circle.radius),
                             int(tmp_radius - circle.radius):int(tmp_radius + circle.radius)
                             ]
        return logo_rotated_image

    def is_logo(self, gray_image: np.ndarray, circle_list: [CircleData],
                method: LogoMethod = LogoMethod.CONTOURS_MATCH) -> [float]:
        """ 判断圆圈是小程序码 logo 的概率
        :param gray_image: np.ndarray
        :param method: LogoMethod
        :param circle_list: list of CircleData
        """
        assert method in LogoMethod

        def get_prob_by_char(_contour_image, _circle) -> float:
            """ 通过验证码识别的方法, 返回 logo 概率 """
            _prob_int = -1
            try:
                # 旋转
                _logo_rotated_image = WechatQRDetector.logo_rotate(contour_image=_contour_image, circle=_circle)

                # 识别
                result_str = pytesseract.image_to_data(
                    _logo_rotated_image, config="-c tessedit_char_whitelist=Ssg985 --psm 10")
                int_list = re.compile(r"\d+").findall(result_str.strip().split("\n")[-1])
                _prob_int = int(int_list[-1])
            except Exception as e:
                print(e)

            if _prob_int == -1:
                return 0
            else:
                return _prob_int / 100

        def get_prob_by_match(_contour_image, _circle) -> float:
            """ 通过图像匹配的方法, 返回 logo 概率 """
            # 加载 logo
            _log_template = self.get_logo_template()
            _logo_contour = WechatQRDetector.image_to_contour(
                cv2.resize(_log_template, (_circle.radius * 2 + 2, _circle.radius * 2 + 2)), thickness=1)

            # logo 裁边
            _w, _h = _logo_contour.shape[::-1]
            _logo_contour = _logo_contour[1:_w - 1, 1:_h - 1]

            # 匹配
            _radius = _circle.radius
            _contour_circle = _contour_image[
                              int(_circle.center_y - _radius):int(_circle.center_y + _radius),
                              int(_circle.center_x - _radius):int(_circle.center_x + _radius)
                              ].copy()

            res = cv2.matchTemplate(_contour_circle, _logo_contour, cv2.TM_CCOEFF_NORMED)
            _prob = 0
            loc = np.where(res >= 0.1)
            if loc is not None:
                for pt in zip(*loc[::-1]):
                    _prob = res[pt[1], pt[0]]
                    break

            return _prob

        def get_prob_by_contour_match(_contour_image, _circle) -> float:
            """ 通过轮廓匹配的方法, 返回 logo 概率 """
            # 加载 logo
            _log_template = self.get_logo_template()
            _logo_contour = WechatQRDetector.image_to_contour(
                cv2.resize(_log_template, (_circle.radius * 2 + 2, _circle.radius * 2 + 2)), thickness=1)

            # logo 裁边
            _w, _h = _logo_contour.shape[::-1]
            _logo_contour = _logo_contour[1:_w - 1, 1:_h - 1]

            # 匹配
            _radius = _circle.radius
            _contour_circle = _contour_image[
                              int(_circle.center_y - _radius):int(_circle.center_y + _radius),
                              int(_circle.center_x - _radius):int(_circle.center_x + _radius)
                              ].copy()

            return cv2.matchShapes(_contour_circle, _logo_contour, cv2.CONTOURS_MATCH_I3, 1.0)

        if method == LogoMethod.CHAR_TESSERACT:
            contour_image = WechatQRDetector.image_to_contour(image=gray_image, is_gray=True, thickness=-1)
        else:
            contour_image = WechatQRDetector.image_to_contour(image=gray_image, is_gray=True, thickness=1)

        # 获取概率
        prob_list = []
        for circle in circle_list:
            if method == LogoMethod.CHAR_TESSERACT:
                prob_list.append(get_prob_by_char(_contour_image=contour_image, _circle=circle))
            elif method == LogoMethod.CONTOURS_MATCH:
                prob_list.append(get_prob_by_contour_match(_contour_image=contour_image, _circle=circle))
            else:
                prob_list.append(get_prob_by_match(_contour_image=contour_image, _circle=circle))

        return prob_list

    def get_logo_template(self, ) -> np.ndarray:
        """ 返回 logo 的模板图片 """
        if self._log_template is None:
            logo_file = os.path.join(os.path.dirname(__file__), "logo.png")
            assert os.path.exists(logo_file)
            self._log_template = cv2.imread(logo_file)

        return self._log_template

    def find_logo_by_size_list(self, gray_image: np.ndarray, size_list: [int], match_threshold: float = 0.2,
                               filter_by_prob: bool = True) -> [CircleData]:
        """使用轮廓图 寻找 logo
        :param gray_image:
        :param size_list:
        :param match_threshold:
        :param filter_by_prob:
        :rtype: list of CircleData
        :return:
        """

        def get_logo_contour(_size: int):
            """ work well """
            _logo_image = self.get_logo_template()
            _logo_contour = WechatQRDetector.image_to_contour(
                cv2.resize(_logo_image, (size + 2, size + 2)), thickness=1)
            _w, _h = _logo_contour.shape[::-1]

            # logo 裁边
            _logo_contour = _logo_contour[1:_w - 1, 1:_h - 1]

            return _logo_contour

        self.logger.info("finding logo...")
        # match_list
        match_result_list = []
        match_contour_image = self.image_to_contour(image=gray_image, is_gray=True, thickness=1)

        for size in size_list:
            logo_contour = get_logo_contour(_size=size)
            w, h = logo_contour.shape[::-1]
            assert w == h and w == size

            res = cv2.matchTemplate(match_contour_image, logo_contour, cv2.TM_CCOEFF_NORMED)
            loc = np.where(res >= match_threshold)

            # 保留前 5 个
            pt_list = []
            for pt in zip(*loc[::-1]):
                prob = res[pt[1], pt[0]]  # 匹配度
                pt_list.append((pt, prob))
            pt_list = sorted(pt_list, key=lambda x: x[1], reverse=True)

            for pt, prob in pt_list[:5]:
                new_circle = CircleData(center_x=int(pt[0] + w / 2), center_y=int(pt[1] + h / 2), radius=int(w / 2))

                _found_index = None
                for _index, (_circle, _prob, _count) in enumerate(match_result_list):
                    if math.sqrt(((_circle.center_x - new_circle.center_x) ** 2 + (
                            _circle.center_y - new_circle.center_y) ** 2)) / new_circle.radius < 0.2:
                        _found_index = _index
                        break

                # 结果更新
                if _found_index is not None:
                    if match_result_list[_found_index][1] < prob:
                        match_result_list[_found_index] = (new_circle, prob, match_result_list[_found_index][2] + 1)
                else:
                    match_result_list.append((new_circle, prob, 1))

        # logo 过滤
        if filter_by_prob:
            prob_list = self.is_logo(gray_image=gray_image,
                                     circle_list=[circle for (circle, prob, count) in match_result_list])
            self.logger.info("prob list is {}".format(prob_list))
            result_list = [circle for index, (circle, prob, count) in enumerate(match_result_list)
                           if prob_list[index] > 0 and count > 0]
        else:
            result_list = [circle for (circle, prob, count) in match_result_list if count > 0]

        # logger
        self.logger.info("{} logo circle found!".format(len(result_list)))
        return result_list

    def find_logo(self, gray_image: np.ndarray, match_threshold: float = 0.2, filter_by_prob: bool = True) \
            -> [CircleData]:
        """使用轮廓图 寻找 logo
        :param gray_image: np.ndarray
        :param match_threshold:
        :param filter_by_prob:
        :rtype: list of CircleData
        :return:
        """
        # 计算尺寸
        image_max_size = min(gray_image.shape[0], gray_image.shape[1])
        min_size = 20
        max_size = int(0.25 * image_max_size)
        step_size = max(5, int((max_size - min_size) / 100))
        if (max_size - min_size) / step_size < 5:
            step_size = 2

        size_list = [size for size in (min_size, max_size, step_size)]

        return self.find_logo_by_size_list(gray_image=gray_image, size_list=size_list,
                                           match_threshold=match_threshold,
                                           filter_by_prob=filter_by_prob)
