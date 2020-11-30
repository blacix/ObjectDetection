import asyncio
import cv2 as cv
from image_processing import ImageProcessor
from camera_calibration import UndistortedVideoCapture
import threading
cap0 = UndistortedVideoCapture(0)
cap2 = UndistortedVideoCapture(2)
# image_processor2 = ImageProcessor(2)
# image_processor0 = ImageProcessor(0)



def task(image):
    print(threading.currentThread().ident)
    # image = image_processor0.process_image(image)
    return image


def main():
    while True:
        print('loop')

        print('reading0')
        ret, image0 = cap0.read()
        print('reading0 done')
        if not ret:
            continue

        print('reading2')
        ret, image2 = cap2.read()
        print('reading2 done')
        if not ret:
            continue

        # cv.imshow('camera 0', image0)
        # cv.imshow('camera 2', image2)
        #
        # image0 = image_processor0.process_image(image0)
        # image2 = image_processor2.process_image(image2)
        #
        cv.imshow('camera 0', image0)
        cv.imshow('camera 2', image2)

        print('loop finished')


        if cv.waitKey(10) == ord('q'):
            break

    # cv.destroyAllWindows()


if __name__ == '__main__':
    main()
