import cv2 as cv

import camera_calibration as calib
import config
from object_detector import ObjectDetector
from marker_detector import MarkerDetector
from motion_detector import MotionDetector


def main():
    cap = calib.UndistortedVideoCapture(config.CAMERA_ID)

    object_detector = ObjectDetector()
    marker_detector = MarkerDetector()
    motion_detector = MotionDetector()

    while True:
        ret, image = cap.read()
        if not ret:
            continue

        image = motion_detector.process(image)
        image = object_detector.detect_objects(image)
        image, _, _ = marker_detector.detect_markers(image)

        cv.imshow('objects', image)
        if cv.waitKey(100) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
