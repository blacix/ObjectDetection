# from object_detector import ObjectDetector
from marker_detector import MarkerDetector
from motion_detector import MotionDetector


class ImageProcessor:
    def __init__(self):
        self.marker_detector = MarkerDetector()
        self.motion_detector = MotionDetector()
        self.image = None
        # self.object_detector = ObjectDetector()

    def process_image(self, image):
        # image = self.object_detector.detect_objects(image)
        self.image = self.motion_detector.process(image)
        self.image, _, _ = self.marker_detector.detect_markers(self.image)
        return self.image
