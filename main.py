import cv2 as cv
from image_processing import ImageProcessor
from camera_calibration import UndistortedVideoCapture
import threading
cap0 = UndistortedVideoCapture(0)
cap2 = UndistortedVideoCapture(2)
image_processor0 = ImageProcessor(0)
image_processor2 = ImageProcessor(2)


def main():
    while True:
        ret, image0 = cap0.read()
        ret, image2 = cap2.read()

        image0 = image_processor0.process_image(image0)
        image2 = image_processor2.process_image(image2)
        cv.imshow('camera 0', image0)
        cv.imshow('camera 2', image2)

        if cv.waitKey(10) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
