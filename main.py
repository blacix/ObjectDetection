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

image_processor0 = ImageProcessor()
image_processor1 = ImageProcessor()

executor = ThreadPoolExecutor(max_workers=3)


def main():
    while True:
        future_elp = executor.submit(process_elp)
        future_logitech = executor.submit(process_logitech)

        image_elp = future_elp.result()
        image_logitech = future_logitech.result()
        cv.imshow('camera elp', image_elp)
        cv.imshow('camera logitech', image_logitech)
        if cv.waitKey(10) == ord('q'):
            break

    cv.destroyAllWindows()


def process_elp():
    print("e")
    ret, image0 = cap_elp.read()
    image0 = image_processor0.process_image(image0)
    return image0


def process_logitech():
    print("l")
    ret, image1 = cap_logitech.read()
    image1 = image_processor1.process_image(image1)
    return image1


if __name__ == '__main__':
    main()
