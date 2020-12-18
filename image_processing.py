from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading
import cv2 as cv
import numpy as np

# from object_detector import ObjectDetector
from marker_detector import MarkerDetector
from motion_detector import MotionDetector


class ImageProcessor:
    def __init__(self):
        self.marker_detector = MarkerDetector()
        self.motion_detector = MotionDetector()
        self.image = None
        # self.object_detector = ObjectDetector()
        self.executor = ThreadPoolExecutor(max_workers=3)

    def process_image(self, image):
        # image = self.object_detector.detect_objects(image)
        self.image, _ = self.motion_detector.process(image)
        self.image, _ = self.marker_detector.process(self.image)
        return self.image

    def process_image_parallel(self, image):
        ret_image = np.zeros(image.shape, np.uint8)

        motion_future = self.executor.submit(self.motion_detector.process, image)
        marker_future = self.executor.submit(self.marker_detector.process, image)

        try:
            motion_image, motion_detected = motion_future.result()
            marker_image, ids = marker_future.result()
        except Exception as exc:
            print(f"exception: {exc}")
        else:
            ret_image = self.decorate_image(image, marker_image)
            ret_image = self.decorate_image(ret_image, motion_image)

        return ret_image

    @staticmethod
    def decorate_image(image, decoration_image):
        # image processors return images that contains only the "highlights" of ROI.
        # simple addition changes colour, but fast
        ret_image = cv.add(image, decoration_image)

        # # bitwise stuff is slow
        # # TODO
        # ret_image = image.copy()
        # image_empty = np.zeros(image.shape, np.uint8)
        # img_gray = cv.cvtColor(decoration_image, cv.COLOR_BGR2GRAY)
        # ret, decoration_mask = cv.threshold(img_gray, 10, 255, cv.THRESH_BINARY)
        # cv.bitwise_and(image, image_empty, ret_image, mask=decoration_mask)
        # cv.bitwise_or(decoration_image, ret_image, ret_image, mask=decoration_mask)

        return ret_image
