import cv2 as cv
from config import CAMERA_ID
import camera_calibration as calib
import numpy as np

IMAGE_AREA_RATIO = 10000


class MotionDetector:
    def __init__(self):
        self.first_frame_gray = None
        self.actual_frame_gray = None
        self.prev_frame_gray = None
        self.idle_counter = 0

    def process(self, image):
        if self.first_frame_gray is None:
            self.first_frame_gray = self.prepare_image(image)

        if self.actual_frame_gray is None:
            self.actual_frame_gray = self.prepare_image(image)

        self.prev_frame_gray = self.actual_frame_gray
        self.actual_frame_gray = self.prepare_image(image)

        image, bounding_boxes = self.get_bounding_boxes(self.prev_frame_gray, self.actual_frame_gray)

        display_text = ""
        if len(list(bounding_boxes)) > 0:
            self.idle_counter = 0
            display_text = "motion"
        else:
            self.idle_counter += 1
            if self.idle_counter > 5:
                self.idle_counter = 5
                display_text = "idle"
            else:
                display_text = "waiting"

        display_text = "boxes: " + str(len(list(bounding_boxes))) + " - " + display_text
        image = cv.putText(image, display_text, (00, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
                           cv.LINE_AA)


    @staticmethod
    def prepare_image(image):
        ret = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        ret = cv.GaussianBlur(ret, (21, 21), 0)
        return ret

    @staticmethod
    def get_bounding_boxes(image1, image2):
        image_with_boxes = image
        bounding_boxes = []

        # compute difference between first frame and current frame
        delta = cv.absdiff(image1, image2)
        ret, thresh_img = cv.threshold(delta, 25, 255, cv.THRESH_BINARY)

        contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
        if contours is not None:
            # cv.drawContours(image, contours, -1, (0, 255, 0), 3)
            for contour in contours:
                image_area = np.prod(image_with_boxes.shape)
                if cv.contourArea(contour) >= image_area / IMAGE_AREA_RATIO:
                    approx = cv.approxPolyDP(contour, 3, True)
                    x, y, w, h = cv.boundingRect(approx)
                    cv.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    bounding_boxes.append((x, y, w, h))

        return image_with_boxes, bounding_boxes


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

        motion_detector.process(image)
        cv.imshow('motion detection', image)
        if cv.waitKey(100) == ord('q'):
            break

    cv.destroyAllWindows()
