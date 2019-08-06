# coding:utf-8
__author__ = 'rk.feng'

import os
import unittest

import cv2

from wechat_qr_detection import WechatQRDetector, LogoMethod

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
    def show(image_path: str, show_gray: bool = False, show_all_circle: bool = False):
        # 载入并显示图片
        img = cv2.imread(image_path)

        # 图片大小上限
        w, h = img.shape[1], img.shape[0]
        max_size = 500
        if w > 1.5 * max_size or h > max_size:
            radio = min(1.5 * max_size / w, max_size / h)
            n_w, n_h = int(w * radio), int(h * radio)
            img = cv2.resize(img, (n_w, n_h))

        detector = WechatQRDetector(img=img)

        # show gray
        if show_gray:
            cv2.imshow("gray", detector.gray)

        # 找到粗略的圆圈
        circle_list = detector.find_circle()
        if show_all_circle:
            img_all = img.copy()
            for circle in circle_list:
                img_all = cv2.circle(
                    img_all, (circle.center_x, circle.center_y), circle.radius, (0, 0, 255), 1, 8, 0
                )
            cv2.imshow("all", img_all)

        # 牛眼
        wechat_info_list = detector.detect_qr_code(circle_list=circle_list)

        # show result
        img_b = img.copy()
        _logo_circle_list = []
        for wechat_qr_info in wechat_info_list:
            img_b = cv2.circle(
                img_b,
                (wechat_qr_info.center_circle.center_x, wechat_qr_info.center_circle.center_y),
                wechat_qr_info.center_circle.radius, (0, 0, 255), 1, 8, 0
            )
            _logo_circle_list.append(wechat_qr_info.logo_circle)

        prob_list = detector.is_logo(
            gray_image=detector.gray, circle_list=_logo_circle_list, method=LogoMethod.IMAGE_MATCH)

        print(prob_list)
        for index, prob in enumerate(prob_list):
            if prob > 0:
                logo_circle = _logo_circle_list[index]
                img_b = cv2.circle(
                    img_b,
                    (logo_circle.center_x, logo_circle.center_y), logo_circle.radius, (0, 0, 255), 1, 8, 0
                )

        # 显示新图像
        cv2.imshow(image_path, img_b)

    def testDemo(self):
        self.show(os.path.join(self.img_dir, 'wx1.png'), show_gray=True, show_all_circle=True)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def testAll(self):
        for image_path in self.list_all_images():
            self.show(image_path=image_path)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def testFindLogo(self):
        """ logo 寻找 """
        image_path = os.path.join(self.img_dir, "wx1.png")
        img = cv2.imread(image_path)
        detector = WechatQRDetector(img=img)

        logo_circle_list = detector.find_logo(gray_image=detector.gray, filter_by_prob=False)

        for circle in logo_circle_list:
            img = cv2.circle(
                img, (circle.center_x, circle.center_y), circle.radius, (0, 0, 255), 1, 8, 0
            )
        cv2.imshow("Detected", img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def testFindLogoAll(self):
        """ logo 寻找 """
        for image_path in self.list_all_images():
            img = cv2.imread(filename=image_path)

            # 图片大小上限
            w, h = img.shape[1], img.shape[0]
            max_size = 500
            if w > 1.5 * max_size or h > max_size:
                radio = min(1.5 * max_size / w, max_size / h)
                n_w, n_h = int(w * radio), int(h * radio)
                img = cv2.resize(img, (n_w, n_h))

            detector = WechatQRDetector(img=img)
            logo_circle_list = detector.find_logo(gray_image=detector.gray, filter_by_prob=True)

            for circle in logo_circle_list:
                img = cv2.circle(
                    img, (circle.center_x, circle.center_y), circle.radius, (0, 0, 255), 1, 8, 0
                )

            cv2.imshow(image_path.split("/")[-1], img)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def testContoursAll(self):
        """ 轮廓图生成效果 测试 """
        for image_path in self.list_all_images():
            img = cv2.imread(filename=image_path)

            # 图片大小上限
            w, h = img.shape[1], img.shape[0]
            max_size = 500
            if w > 1.5 * max_size or h > max_size:
                radio = min(1.5 * max_size / w, max_size / h)
                n_w, n_h = int(w * radio), int(h * radio)
                img = cv2.resize(img, (n_w, n_h))

            contour_image = WechatQRDetector.image_to_contour(image=img)
            cv2.imshow(image_path.split("/")[-1], contour_image)

        cv2.waitKey(0)
        cv2.destroyAllWindows()

