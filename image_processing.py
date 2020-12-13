import cv2 as cv
from camera_calibration import UndistortedVideoCapture
# from object_detector import ObjectDetector
from marker_detector import MarkerDetector
from motion_detector import MotionDetector
from threading import Thread
import threading
import time


class ImageProcessor:
    def __init__(self):
        self.marker_detector = MarkerDetector()
        self.motion_detector = MotionDetector()
        self.thread = None
        self.image = None
        # self.object_detector = ObjectDetector()
    #     self.camera_id = camera_id
    #     self.cap = None # UndistortedVideoCapture(self.camera_id)
    #     self.thread = Thread(target=self.loop)
    #     self.thread.start()
    #
    # def loop(self):
    #     self.cap = UndistortedVideoCapture(self.camera_id)
    #     while True:
    #         print(threading.currentThread().ident)
    #         ret, image = self.cap.read()
    #         image = self.process_image(image)
    #         time.sleep(10)
    #         cv.imshow('camera {}'.format(self.camera_id), image)
    #         if cv.waitKey(100) == ord('q'):
    #             break

    def process_image(self, image):
        # image = self.object_detector.detect_objects(image)
        self.image = self.motion_detector.process(image)
        self.image, _, _ = self.marker_detector.detect_markers(self.image)
        return self.image

