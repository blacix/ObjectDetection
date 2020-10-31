import cv2 as cv
import cv2.aruco as aruco
from config import CAMERA_ID
import camera_calibration as calib


class MarkerDetector:
    def __init__(self):
        self.aruco_dict = aruco.Dictionary_get(aruco.DICT_ARUCO_ORIGINAL)
        # print(aruco_dict)
        self.arucoParameters = aruco.DetectorParameters_create()
        self.arucoParameters.cornerRefinementMethod = cv.aruco.CORNER_REFINE_SUBPIX
        # self.arucoParameters.minDistanceToBorder = 5

    def detect_markers(self, image):
        gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        corners, ids, rejected_img_points = aruco.detectMarkers(
            gray, self.aruco_dict, parameters=self.arucoParameters)
        image_markers = aruco.drawDetectedMarkers(image, corners, ids)
        return image_markers, corners, ids


if __name__ == '__main__':
    cap = calib.UndistortedVideoCapture(CAMERA_ID)
    marker_detector = MarkerDetector()

    while True:
        ret, image = cap.read()
        if not ret:
            continue

        image, _, _ = marker_detector.detect_markers(image)
        cv.imshow("markers", image)
        if cv.waitKey(100) == ord('q'):
            break

    cv.destroyAllWindows()
