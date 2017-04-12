from __future__ import absolute_import  # fix known bug of PyCharm
import unittest
import cv2
from src.raw_photo import RawPhoto


class MyTestCase(unittest.TestCase):

    def test_normal0(self):
        test_img = cv2.imread("tst/test0.jpeg", 0)
        rp = RawPhoto(test_img, 2, 30)
        res = rp.dump_data()
        rp.paper_objs = []
        print(res)

    def test_normal1(self):
        test_img = cv2.imread("tst/test1.png", 0)
        rp = RawPhoto(test_img, 2, 30)
        res = rp.dump_data()
        rp.paper_objs = []
        print(res)

    def test_normal2(self):
        test_img = cv2.imread("tst/test2.jpeg", 0)
        rp = RawPhoto(test_img, 2, 30)
        res = rp.dump_data()
        rp.paper_objs = []
        print(res)


if __name__ == '__main__':
    unittest.main()
