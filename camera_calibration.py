#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv

from config import CAMERA_ID, USE_CALIBRATION


class UndistortedVideoCapture:
    def __init__(self, camera_id):
        self.camera_calibration = CameraCalibration()
        self.camera_calibration.load_calibration()
        self.video_capture = cv.VideoCapture(camera_id)

    def read(self):
        ret, image = self.video_capture.read()
        if ret:
            if USE_CALIBRATION:
                image = self.camera_calibration.undistort_image(image)
        return ret, image


class CameraCalibration:
    def __init__(self):
        self.camera_matrix = None
        self.new_camera_matrix = None
        self.dist_coeffs = None

    def calibrate_camera(self):
        square_size = 5.0
        pattern_size = (9, 6)
        # prepare pattern_points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        pattern_points = np.zeros((np.prod(pattern_size), 3), np.float32)
        pattern_points[:, :2] = np.indices(pattern_size).T.reshape(-1, 2)
        pattern_points *= square_size

        obj_points = []
        img_points = []

        term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_COUNT, 30, 0.1)

        cap = cv.VideoCapture(CAMERA_ID)
        while True:
            ret, img = cap.read()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCorners(img, pattern_size)

            if found:
                obj_points.append(pattern_points)

                corners_subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), term_criteria)
                img_points.append(corners_subpix)

                # Draw and display the corners
                img = cv.drawChessboardCorners(img, pattern_size, corners_subpix, ret)

            cv.imshow('calibration', img)
            if cv.waitKey(500) == ord('q'):
                break;

        cv.destroyWindow('calibration')
        h, w, _ = img.shape
        h, w, = gray.shape

        print("calibrating camera...");
        if len(obj_points) > 0:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None,
                                                                               None)
            # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

            print("calculating optimal camera matrix...");
            new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

        print("camera  calibrated")
        cv.destroyWindow('calibration')

    def save_calibration(self):
        print('Saving calibration...')

        # writes array to .yml file
        fs_write = cv.FileStorage('calibration.yml', cv.FILE_STORAGE_WRITE)
        arr = np.random.rand(5, 5)
        fs_write.write("camera_matrix", self.camera_matrix)
        fs_write.write("new_camera_matrix", self.new_camera_matrix)
        fs_write.write("dist_coeffs", self.dist_coeffs)
        fs_write.release()

    def load_calibration(self):
        print('Loading calibration...')
        fs_read = cv.FileStorage('calibration.yml', cv.FILE_STORAGE_READ)
        self.camera_matrix = fs_read.getNode('camera_matrix').mat()
        self.new_camera_matrix = fs_read.getNode('new_camera_matrix').mat()
        self.dist_coeffs = fs_read.getNode('dist_coeffs').mat()

        fs_read.release()

    def undistort_image(self, img):
        if not self.is_camera_calibrated():
            return img

        # with undistort
        dst = cv.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)

        # with remap
        # mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv.CV_32FC1)
        # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]

        return dst

    def is_camera_calibrated(self):
        return self.camera_matrix is not None and self.new_camera_matrix is not None and self.dist_coeffs is not None


def main():
    camera_calibration = CameraCalibration()
    camera_calibration.calibrate_camera()
    camera_calibration.save_calibration()
    # camera_calibration.load_calibration()
    cap = UndistortedVideoCapture(CAMERA_ID)
    while True:
        ret, img = cap.read()
        if not ret:
            continue

        cv.imshow("calibration", img)
        if cv.waitKey(100) == ord('q'):
            break

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
