import asyncio
import cv2 as cv
from image_processing import ImageProcessor
from camera_calibration import UndistortedVideoCapture
import threading
cap0 = UndistortedVideoCapture(1, fisheye=True)
image_processor0 = ImageProcessor()


def task(image):
    print(threading.currentThread().ident)
    # image = image_processor0.process_image(image)
    return image


def main():
    while True:
        ret, image0 = cap0.read()
        if not ret:
            continue

        cv.imshow('camera 0', image0)
        image0 = image_processor0.process_image(image0)
        cv.imshow('camera 0', image0)

        if cv.waitKey(10) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
