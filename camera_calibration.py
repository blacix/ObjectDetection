#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv


def calibrate_camera():
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

    cap = cv.VideoCapture(1)
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

    h, w, _ = img.shape
    h, w, = gray.shape

    print("calibrating camera...");
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None, None)
    # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

    print("calculating optimal camera matrix...");
    new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

    print("camera  calibrated")
    cv.destroyWindow('calibration')

    return camera_matrix, new_camera_matrix, dist_coeffs


def undistort_image(img, camera_matrix, new_camera_matrix, dist_coeffs):
    # with undistort
    dst = cv.undistort(img, camera_matrix, dist_coeffs, None, new_camera_matrix)

    # with remap
    # mapx, mapy = cv.initUndistortRectifyMap(camera_matrix, dist_coeffs, None, new_camera_matrix, (w, h), cv.CV_32FC1)
    # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

    # crop the image
    # x, y, w, h = roi
    # dst = dst[y:y + h, x:x + w]

    return dst


def main():
    camera_matrix, new_camera_matrix, dist_coeffs = calibrate_camera()

    cap = cv.VideoCapture(1)
    while True:
        ret, img = cap.read()
        undistorted_image = undistort_image(img, camera_matrix, new_camera_matrix, dist_coeffs)
        cv.imshow('undistorted', undistorted_image)
        if cv.waitKey(500) == ord('q'):
            break;

    cv.destroyAllWindows()


if __name__ == '__main__':
    main()
