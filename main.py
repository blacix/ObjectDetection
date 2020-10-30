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
        undistorted_image = calib.undistort_image(image)

        image_objects = object_detector.detect_objects(undistorted_image)
        cv.imshow('images', image_objects)

        image_markers, _, _ = marker_detector.detect_markers(undistorted_image)
        cv.imshow('markers', image_markers)

        if cv.waitKey(100) == ord('q'):
            break


if __name__ == '__main__':
    main()
