# coding:utf-8
__author__ = 'rk.feng'

import os
import unittest

import cv2

from wechat_qr_detection import WechatQRDetector

_cur_dir = os.path.dirname(__file__)


class TestDetection(unittest.TestCase):
    def setUp(self) -> None:
        self.img_dir = os.path.join(_cur_dir, "wx_qr")

    def list_all_images(self, ):
        file_list = [os.path.join(self.img_dir, _file) for _file in os.listdir(self.img_dir)]
        image_list = []
        for _file in file_list:
            if _file.endswith("png") or _file.endswith(".jpg"):
                image_list.append(_file)

        return image_list

    @staticmethod
    def show(image_path: str):
        # 载入并显示图片
        img = cv2.imread(image_path)

        detector = WechatQRDetector(img=img)

        # 找到粗略的圆圈
        circle_list = detector.find_circle()

        # # 牛眼
        wechat_info_list = detector.detect_qr_code(circle_list=circle_list)

        img_b = img.copy()
        for wechat_qr_info in wechat_info_list:
            img_b = cv2.circle(
                img_b,
                (wechat_qr_info.center_circle.center_x, wechat_qr_info.center_circle.center_y),
                wechat_qr_info.center_circle.radius, (0, 0, 255), 1, 8, 0
            )

        # 显示新图像
        cv2.imshow(image_path, img_b)

    def testDemo(self):
        self.show(os.path.join(self.img_dir, 'wx12.png'))
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def testAll(self):
        for image_path in self.list_all_images():
            self.show(image_path=image_path)

        cv2.waitKey(0)
        cv2.destroyAllWindows()
