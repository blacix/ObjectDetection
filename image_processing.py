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
        # print(f'process_image_async {threading.currentThread().ident}')
        # TODO all must have the same interface: .process(image)
        arr = [self.motion_detector, self.marker_detector]
        ret_image = image.copy()
        future_to_processors = \
            {self.executor.submit(p.process, image): p for p in arr}
        for future in concurrent.futures.as_completed(future_to_processors):
            p = future_to_processors[future]
            try:
                processed_image, _ = future.result()
            except Exception as exc:
                print(f"exception: {exc}")
            else:
                ret_image = cv.add(ret_image, processed_image)
        return ret_image
