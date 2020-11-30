import cv2 as cv

import camera_calibration as calib
import config
from object_detector import ObjectDetector
from marker_detector import MarkerDetector
from motion_detector import MotionDetector


class ImageProcessor:
    def __init__(self, camera_id):
        self.marker_detector = MarkerDetector()
        self.motion_detector = MotionDetector()
        self.object_detector = ObjectDetector()
        self.camera_id = camera_id
        self.cap = calib.UndistortedVideoCapture(self.camera_id)

    def process_image(self):
        ret, image = self.cap.read()
        # if not ret:
        #     return image

        # image = self.motion_detector.process(image)
        image = self.object_detector.detect_objects(image)
        image, _, _ = self.marker_detector.detect_markers(image)
        return image
