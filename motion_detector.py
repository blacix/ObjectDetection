import cv2 as cv
from config import CAMERA_ID
import camera_calibration as calib
import numpy as np
from enum import Enum, auto


IMAGE_AREA_RATIO = 10000


class State(Enum):
    IDLE = auto()
    WAITING = auto()
    MOTION = auto()


class MotionDetector:
    def __init__(self):
        self.first_frame = None
        self.actual_frame = None
        self.prev_frame = None
        self.idle_counter = 0
        self.motion = True
        self.state = State.IDLE

    def process(self, image):
        if self.first_frame is None:
            self.first_frame = image.copy()

        if self.actual_frame is None:
            self.actual_frame = image.copy()

        self.prev_frame = self.actual_frame.copy()
        self.actual_frame = image.copy()

        hist_compare_result = self.compare_hist(self.prev_frame, self.actual_frame)
        image_with_bbs, bounding_boxes = self.get_bounding_boxes(self.prev_frame, self.actual_frame)

        if len(list(bounding_boxes)) > 0:
            self.motion = True
            self.idle_counter = 0
            self.state = State.MOTION
        else:
            self.idle_counter += 1
            if self.idle_counter > 5:
                self.motion = False
                self.idle_counter = 5
                self.state = State.IDLE
            else:
                self.motion = True
                self.state = State.WAITING

        # display_text = "dHist: " + str(hist_compare_result)[0:5] + " boxes: " + str(len(list(bounding_boxes))) \
        #                + " - " + state
        # image_with_bbs = cv.putText(image, display_text, (00, 450), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2,
        #                             cv.LINE_AA)

        return image_with_bbs, self.motion

    @staticmethod
    def prepare_image(image):
        prepared_image = image.copy()
        prepared_image = cv.cvtColor(prepared_image, cv.COLOR_BGR2GRAY)
        prepared_image = cv.GaussianBlur(prepared_image, (21, 21), 0)
        return prepared_image

    @staticmethod
    def get_bounding_boxes(image1, image2):
        image1_gray = MotionDetector.prepare_image(image1)
        image2_gray = MotionDetector.prepare_image(image2)
        # image_with_boxes = image2.copy()
        image_with_boxes = np.zeros(image2.shape, np.uint8)
        bounding_boxes = []

        # compute difference between first frame and current frame
        delta = cv.absdiff(image1_gray, image2_gray)
        ret, thresh_img = cv.threshold(delta, 25, 255, cv.THRESH_BINARY)
        # thresh_img = cv.dilate(thresh_img, None, iterations=2)

        contours, hierarchy = cv.findContours(thresh_img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_TC89_L1)
        if contours is not None:
            # cv.drawContours(image_with_boxes, contours, -1, (0, 255, 0), 3)
            for contour in contours:
                image_area = np.prod(image_with_boxes.shape)
                if cv.contourArea(contour) >= 1:  # image_area / IMAGE_AREA_RATIO:
                    approx = cv.approxPolyDP(contour, 3, True)
                    x, y, w, h = cv.boundingRect(approx)
                    cv.rectangle(image_with_boxes, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    bounding_boxes.append((x, y, w, h))

        return image_with_boxes, bounding_boxes

    @staticmethod
    def compare_hist(image1, image2):
        h_bins = 50
        s_bins = 60

        hist_size = [h_bins, s_bins]
        channels = [0, 1]
        # hue varies from 0 to 179, saturation from 0 to 255
        h_ranges = [0, 180]
        s_ranges = [0, 256]
        ranges = [0, 180, 0, 256]

        # Convert to HSV format
        hvsImage1 = cv.cvtColor(image1, cv.COLOR_BGR2HSV)
        hvsImage2 = cv.cvtColor(image2, cv.COLOR_BGR2HSV)

        mask = None
        hist1 = cv.calcHist([hvsImage1], channels, mask, hist_size, ranges)
        cv.normalize(hist1, hist1, 0, 1, cv.NORM_MINMAX)

        hist2 = cv.calcHist([hvsImage2], channels, mask, hist_size, ranges)
        cv.normalize(hist2, hist2, 0, 1, cv.NORM_MINMAX)

        result = cv.compareHist(hist1, hist2, cv.HISTCMP_CHISQR)
        return result


if __name__ == '__main__':
    cap = calib.UndistortedVideoCapture(CAMERA_ID)
    motion_detector = MotionDetector()

    # wait for auto focus
    for i in range(100):
        ret, frame = cap.read()

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame, _ = motion_detector.process(frame)
        cv.imshow('motion detection', frame)
        if cv.waitKey(100) == ord('q'):
            break

    cv.destroyAllWindows()
