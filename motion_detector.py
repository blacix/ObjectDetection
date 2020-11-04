import cv2 as cv
from config import CAMERA_ID
import camera_calibration as calib
import numpy as np


class MotionDetector:
    def __init__(self):
        self.first_frame_gray = None
        self.actual_frame_gray = None

    def detect_motion(self, image):
        if self.first_frame_gray is None:
            self.first_frame_gray = self.prepare_image(image)
            return self.first_frame_gray

        self.actual_frame_gray = self.prepare_image(image)

        # compute difference between first frame and current frame
        delta = cv.absdiff(self.first_frame_gray, self.actual_frame_gray)
        ret, thresh_img = cv.threshold(delta, 25, 255, cv.THRESH_BINARY)

        contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
        if contours is not None:
            # cv.drawContours(image, contours, -1, (0, 255, 0), 3)
            for contour in contours:
                image_area = np.prod(image.shape)
                if cv.contourArea(contour) >= image_area / 500:
                    approx = cv.approxPolyDP(contour, 3, True)
                    x, y, w, h = cv.boundingRect(approx)
                    cv.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)

        return image

    @staticmethod
    def prepare_image(image):
        ret = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret = cv.GaussianBlur(ret, (21, 21), 0)
        return ret


if __name__ == '__main__':
    cap = calib.UndistortedVideoCapture(CAMERA_ID)
    motion_detector = MotionDetector()

    # wait for auto focus
    for i in range(100):
        ret, image = cap.read()

    while True:
        ret, image = cap.read()
        if not ret:
            continue

        image = motion_detector.detect_motion(image)
        cv.imshow('motion detection', image)
        if cv.waitKey(100) == ord('q'):
            break

    cv.destroyAllWindows()
