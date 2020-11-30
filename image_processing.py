import cv2 as cv
from camera_calibration import UndistortedVideoCapture
# from object_detector import ObjectDetector
from marker_detector import MarkerDetector
from motion_detector import MotionDetector
from threading import Thread
import threading
import time


class ImageProcessor:
    def __init__(self, camera_id):
        self.marker_detector = MarkerDetector()
        self.motion_detector = MotionDetector()
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
        # if not ret:
        #     return image

        # image = self.object_detector.detect_objects(image)
        image = self.motion_detector.process(image)
        image, _, _ = self.marker_detector.detect_markers(image)
        return image
