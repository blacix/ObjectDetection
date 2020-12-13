import cv2 as cv
from image_processing import ImageProcessor
from camera_calibration import UndistortedVideoCapture
import threading
from threading import Thread
import time
from concurrent.futures import ThreadPoolExecutor


cap_elp = UndistortedVideoCapture(0, fisheye=True)
cap_elp.camera_calibration.load_calibration('elp')
cap_logitech = UndistortedVideoCapture(2, fisheye=False)
cap_logitech.camera_calibration.load_calibration('logitech')

image_processor_elp = ImageProcessor()
image_processor_logitech = ImageProcessor()

executor = ThreadPoolExecutor(max_workers=3)


def main():
    while True:
        future_elp = executor.submit(process, image_processor_elp, cap_elp)
        future_logitech = executor.submit(process, image_processor_logitech, cap_logitech)

        image_elp = future_elp.result()
        image_logitech = future_logitech.result()
        cv.imshow('camera elp', image_elp)
        cv.imshow('camera logitech', image_logitech)
        if cv.waitKey(10) == ord('q'):
            break

    cv.destroyAllWindows()


def process(image_processor, cap):
    ret, image = cap.read()
    image = image_processor.process_image(image)
    return image


if __name__ == '__main__':
    main()
