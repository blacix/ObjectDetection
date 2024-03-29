import cv2 as cv
import cv2.aruco as aruco
import numpy as np
from config import CAMERA_ID
import camera_calibration as calib

MARKER_DICT_SIZE = 50


class MarkerDetector:
    def __init__(self):
        # self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        self.aruco_dict = aruco.Dictionary_create(MARKER_DICT_SIZE, 3)
        # print(aruco_dict)
        self.arucoParameters = aruco.DetectorParameters_create()
        self.arucoParameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        # self.arucoParameters.minDistanceToBorder = 5

    def process(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.arucoParameters)
        image_markers = np.zeros(image.shape, np.uint8)
        image_markers = aruco.drawDetectedMarkers(image_markers, corners, ids)
        if ids is None:
            ids = []
        return image_markers, ids

    def generate_marker_images(self):
        self.aruco_dict = aruco.Dictionary_create(MARKER_DICT_SIZE, 3)
        for i in range(MARKER_DICT_SIZE):
            marker_image = cv.aruco.drawMarker(self.aruco_dict, i, 200)
            file_name = 'aruco_3x3_50_'
            if i < 10:
                file_name = file_name + '0'
            file_name = file_name + str(i) + '.jpg'
            cv.imwrite(file_name, marker_image)


if __name__ == '__main__':
    cap = calib.UndistortedVideoCapture(CAMERA_ID)
    marker_detector = MarkerDetector()
    # marker_detector.generate_marker_images()

    marker_count = 0
    while True:
        ret, image = cap.read()
        if not ret:
            continue
        image, ids = marker_detector.process(image)
        if ids is not None:
            marker_count = len(ids)
        text = "markers: {}".format(marker_count)
        image = cv.putText(image, text, (00, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        cv.imshow("markers", image)
        if cv.waitKey(100) == ord('q'):
            break

    cv.destroyAllWindows()
