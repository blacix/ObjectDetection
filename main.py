import asyncio
import cv2 as cv
from image_processing import ImageProcessor
from camera_calibration import UndistortedVideoCapture
import threading
cap0 = UndistortedVideoCapture(2, fisheye=True)
cap1 = UndistortedVideoCapture(0, fisheye=False)
cap1.camera_calibration.load_calibration('logitech')

image_processor0 = ImageProcessor()
image_processor1 = ImageProcessor()


def task(image):
    print(threading.currentThread().ident)
    # image = image_processor0.process_image(image)
    return image


def main():
    while True:
        ret, image0 = cap0.read()
        if not ret:
            continue
        image0 = image_processor0.process_image(image0)
        cv.imshow('camera 0', image0)
        if cv.waitKey(10) == ord('q'):
            break

        # ret, image1 = cap1.read()
        # if not ret:
        #     continue
        # image1 = image_processor1.process_image(image1)
        # cv.imshow('camera 1', image1)
        # if cv.waitKey(10) == ord('q'):
        #     break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
