from concurrent.futures import ThreadPoolExecutor
import concurrent.futures
import threading

# from object_detector import ObjectDetector
from marker_detector import MarkerDetector
from motion_detector import MotionDetector


class ImageProcessor:
    def __init__(self):
        self.marker_detector = MarkerDetector()
        self.motion_detector = MotionDetector()
        self.image = None
        # self.object_detector = ObjectDetector()
        # self.executor = ThreadPoolExecutor(max_workers=3)

    def process_image(self, image):
        print(threading.current_thread().name)
        # print(threading.get_ident())
        # image = self.object_detector.detect_objects(image)
        self.image = self.motion_detector.process(image)
        self.image, _, _ = self.marker_detector.process(self.image)
        return self.image

    # async def process_image_async(self, image):
    #     future = self.executor.submit(self.process_image, image)
    #     return future.result()

    # async def process_image_async(self, image):
    #     arr = [self.marker_detector, self.motion_detector]
    #     images = []
    #     future_to_processors = \
    #         {self.executor.submit(p.process, image): p for p in arr}
    #     for future in concurrent.futures.as_completed(future_to_processors):
    #         p = future_to_processors[future]
    #         try:
    #             images.append(future.result())
    #             # image = future.result()
    #         except Exception as exc:
    #             print(f"exception: {exc}")
    #         else:
    #             pass
    #
    #     return images
