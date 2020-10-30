import cv2 as cv

import camera_calibration as calib
import config
from object_detector import ObjectDetector
from marker_detector import MarkerDetector


def main():
    cap = cv.VideoCapture(config.CAMERA_ID)

    calib.load_calibration()

    object_detector = ObjectDetector()
    marker_detector = MarkerDetector()

    while True:
        ret, image = cap.read()
        image = calib.undistort_image(image)

        image = object_detector.detect_objects(image)
        image, _, _ = marker_detector.detect_markers(image)
        cv.imshow('objects', image)
        if cv.waitKey(100) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
