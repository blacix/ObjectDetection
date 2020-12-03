#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import glob

from config import CAMERA_ID, USE_CALIBRATION

class CameraCalibration:
    def __init__(self, camera_id, fisheye=False):
        self.camera_id = camera_id
        self.fisheye = fisheye
        self.camera_matrix = None
        self.new_camera_matrix = None
        self.dist_coeffs = None
        self.obj_points = []  # 3d point in real world space
        self.img_points = []  # 2d points in image plane
        self.pattern_points = None
        self.shape = None
        self.DIM = None
        self.dim1 = None
        self.dim2 = None
        self.dim3 = None
        self.scaled_K = None


        self.checkboard_pattern_size = (9, 6)
        self.term_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        # self.subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)

        self.fisheye_find_corners_flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_FAST_CHECK + cv.CALIB_CB_NORMALIZE_IMAGE

        # (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6)
        self.fisheye_calibration_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
        self.fisheye_calibration_flags = \
            cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC + cv.fisheye.CALIB_FIX_SKEW + cv.fisheye.CALIB_CHECK_COND

    def calibrate(self):
        if self.fisheye:
            self.pattern_points = np.zeros((1, self.checkboard_pattern_size[0] * self.checkboard_pattern_size[1], 3), np.float32)
            self.pattern_points[0, :, :2] = np.mgrid[0:self.checkboard_pattern_size[0], 0:self.checkboard_pattern_size[1]].T.reshape(-1, 2)
        else:
            square_size = 5.0
            # prepare pattern_points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
            self.pattern_points = np.zeros((np.prod(self.checkboard_pattern_size), 3), np.float32)
            self.pattern_points[:, :2] = np.indices(self.checkboard_pattern_size).T.reshape(-1, 2)
            self.pattern_points *= square_size

        self.process_images()

        if self.fisheye:
            self.calibrate_fisheye_camera()
        else:
            self.calibrate_camera()

        print("K=np.array(" + str(self.camera_matrix.tolist()) + ")")
        print("new K=np.array(" + str(self.new_camera_matrix.tolist()) + ")")
        print("D=np.array(" + str(self.dist_coeffs.tolist()) + ")")

    def process_images(self):
        cap = cv.VideoCapture(self.camera_id)
        while True:
            ret, img = cap.read()
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            found, corners = cv.findChessboardCorners(img, self.checkboard_pattern_size)
            if found:
                self.obj_points.append(self.pattern_points)
                corners_subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.term_criteria)
                self.img_points.append(corners_subpix)

                # Draw and display the corners
                img = cv.drawChessboardCorners(img, self.checkboard_pattern_size, corners_subpix, ret)

            cv.imshow('calibration', img)
            if cv.waitKey(500) == ord('q'):
                break

        self.shape = img.shape
        cv.destroyWindow('calibration')

    def calibrate_fisheye_camera(self):
        h, w, _ = self.shape
        DIM = self.shape[:2][::-1]  # (w, h)
        N_OK = len(self.obj_points)

        print("Found " + str(N_OK) + " valid images for calibration")
        print("DIM=" + str(DIM))

        self.camera_matrix = np.zeros((3, 3))
        self.dist_coeffs = np.zeros((4, 1))

        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        rms, _, _, _, _ = \
            cv.fisheye.calibrate(
                self.obj_points,
                self.img_points,
                DIM,  # (w, h)
                self.camera_matrix,
                self.dist_coeffs,
                rvecs,
                tvecs,
                self.fisheye_calibration_flags,
                self.fisheye_calibration_criteria
            )

        # calculating new camera matrix
        balance = 1
        self.dim1 = DIM  # self.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
        assert self.dim1[0] / self.dim1[1] == DIM[0] / DIM[1], \
            "Image to undistort needs to have same aspect ratio as the ones used in calibration"
        if not self.dim2:
            self.dim2 = self.dim1
        if not self.dim3:
            self.dim3 = self.dim1

        self.scaled_K = self.camera_matrix * self.dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
        self.scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

        # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image.
        self.new_camera_matrix = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(self.scaled_K, self.dist_coeffs,
                                                                                       self.dim2,
                                                                                       np.eye(3), balance=balance)

    def calibrate_camera(self):
        h, w, _ = self.shape

        print("calibrating camera...")
        if len(self.obj_points) > 0:
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(self.obj_points, self.img_points, (w, h),
                                                                               None, None)
            # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)

            print("calculating optimal camera matrix...");
            new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))

            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.new_camera_matrix = new_camera_matrix

    def undistort(self, image):
        if not self.is_camera_calibrated():
            return image

        if self.fisheye:
            return self.undistort_fisheye_image(image)
        else:
            return self.undistort_normal_image(image)

    def undistort_fisheye_image(self, image):
        # with remap
        map1, map2 = cv.fisheye.initUndistortRectifyMap(self.scaled_K, self.dist_coeffs, np.eye(3),
                                                        self.new_camera_matrix, self.dim3, cv.CV_16SC2)
        undistorted_img = cv.remap(image, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

        # undistorted_img = cv.fisheye.undistortImage(image, self.camera_matrix, self.dist_coeffs, None,
        #                                             self.new_camera_matrix)
        return undistorted_img

    def undistort_image(self, img):
        # with undistort
        undistorted_image = cv.undistort(img, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)

        # with remap
        # h, w, _ = self.shape
        # mapx, mapy = cv.initUndistortRectifyMap(self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix, (w, h), cv.CV_32FC1)
        # dst = cv.remap(img, mapx, mapy, cv.INTER_LINEAR)

        # crop the image
        # x, y, w, h = roi
        # dst = dst[y:y + h, x:x + w]

        return undistorted_image

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

    def is_camera_calibrated(self):
        return self.camera_matrix is not None and self.new_camera_matrix is not None and self.dist_coeffs is not None

    # def calibrate_fisheye_old(self):
    #     # https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
    #
    #     pattern_points = np.zeros((1, self.checkboard_pattern_size[0] * self.checkboard_pattern_size[1], 3), np.float32)
    #     pattern_points[0, :, :2] = np.mgrid[0:self.checkboard_pattern_size[0], 0:self.checkboard_pattern_size[1]].T.reshape(-1, 2)
    #     _img_shape = None
    #     obj_points = []  # 3d point in real world space
    #     img_points = []  # 2d points in image plane.
    #     images = glob.glob('*.jpg')
    #
    #     cap = cv.VideoCapture(self.camera_id)
    #     while True:
    #         ret, img = cap.read()
    #         if not ret:
    #             continue
    #
    #         print(img.shape)
    #         _img_shape = img.shape[:2]
    #         print(img.shape[:2])
    #         print(img.shape[::-1])
    #         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #         # Find the chess board corners
    #         ret, corners = cv.findChessboardCorners(gray, self.checkboard_pattern_size, self.fisheye_find_corners_flags)
    #         # If found, add object points, image points (after refining them)
    #         if ret:
    #             obj_points.append(pattern_points)
    #             cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), self.subpix_criteria)
    #             img_points.append(corners)
    #
    #         img = cv.drawChessboardCorners(img, (9, 6), corners, ret)
    #         cv.imshow('calibration', img)
    #         if cv.waitKey(500) == ord('q'):
    #             break
    #
    #     N_OK = len(obj_points)
    #     self.camera_matrix = np.zeros((3, 3))
    #     self.dist_coeffs = np.zeros((4, 1))
    #     rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    #     tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
    #     print('!!!!!!!')
    #     print(gray.shape[::-1])
    #     rms, _, _, _, _ = \
    #         cv.fisheye.calibrate(
    #             obj_points,
    #             img_points,
    #             gray.shape[::-1],
    #             self.camera_matrix,
    #             self.dist_coeffs,
    #             rvecs,
    #             tvecs,
    #             self.fisheye_calibration_flags,
    #             self.fisheye_calibration_criteria
    #         )
    #
    #     print("Found " + str(N_OK) + " valid images for calibration")
    #     print("DIM=" + str(_img_shape[::-1]))
    #     print("K=np.array(" + str(self.camera_matrix.tolist()) + ")")
    #     print("D=np.array(" + str(self.dist_coeffs.tolist()) + ")")
    #
    #     # calculating new camera matrix
    #     DIM = _img_shape[::-1]
    #     balance = 1
    #     dim2 = None
    #     dim3 = None
    #
    #     dim1 = img.shape[:2][::-1]  # dim1 is the dimension of input image to un-distort
    #     assert dim1[0] / dim1[1] == DIM[0] / DIM[
    #         1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    #     if not dim2:
    #         dim2 = dim1
    #     if not dim3:
    #         dim3 = dim1
    #     scaled_K = self.camera_matrix * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.
    #     scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0
    #
    #     # This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
    #     self.new_camera_matrix = cv.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, self.dist_coeffs, dim2, np.eye(3), balance=balance)
    #
    #
    #
    #     print("calibration done")
    #     while True:
    #         ret, img = cap.read()
    #         if not ret:
    #             continue
    #
    #
    #         # with remap
    #         # map1, map2 = cv2.fisheye.initUndistortRectifyMap(scaled_K, D, np.eye(3), new_K, dim3, cv2.CV_16SC2)
    #         # undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    #
    #         undistorted_img = cv.fisheye.undistortImage(img, self.camera_matrix, self.dist_coeffs, None, self.new_camera_matrix)
    #
    #         cv.imshow('calibration', undistorted_img)
    #         if cv.waitKey(500) == ord('q'):
    #             break
    #
    # def calibrate_camera_old(self):
    #     square_size = 5.0
    #     # prepare pattern_points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
    #     pattern_points = np.zeros((np.prod(self.checkboard_pattern_size), 3), np.float32)
    #     pattern_points[:, :2] = np.indices(self.checkboard_pattern_size).T.reshape(-1, 2)
    #     pattern_points *= square_size
    #
    #     obj_points = []
    #     img_points = []
    #
    #     cap = cv.VideoCapture(self.camera_id)
    #     while True:
    #         ret, img = cap.read()
    #         gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
    #         found, corners = cv.findChessboardCorners(img, self.checkboard_pattern_size)
    #
    #         if found:
    #             obj_points.append(pattern_points)
    #             corners_subpix = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), self.term_criteria)
    #             img_points.append(corners_subpix)
    #
    #             # Draw and display the corners
    #             img = cv.drawChessboardCorners(img, self.checkboard_pattern_size, corners_subpix, ret)
    #
    #         cv.imshow('calibration', img)
    #         if cv.waitKey(500) == ord('q'):
    #             break
    #
    #     cv.destroyWindow('calibration')
    #     h, w, _ = gray.shape
    #
    #     print("calibrating camera...")
    #     if len(obj_points) > 0:
    #         ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, (w, h), None,
    #                                                                            None)
    #         # ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    #
    #         print("calculating optimal camera matrix...");
    #         new_camera_matrix, roi = cv.getOptimalNewCameraMatrix(camera_matrix, dist_coeffs, (w, h), 1, (w, h))
    #
    #         self.camera_matrix = camera_matrix
    #         self.dist_coeffs = dist_coeffs
    #         self.new_camera_matrix = new_camera_matrix
    #
    #     print("camera  calibrated")
    #     cv.destroyWindow('calibration')


class UndistortedVideoCapture:
    def __init__(self, camera_id, camera_calibration=None):
        if camera_calibration is None:
            self.camera_calibration = CameraCalibration(camera_id)
            self.camera_calibration.load_calibration()
        else:
            self.camera_calibration = camera_calibration
        self.video_capture = cv.VideoCapture(camera_id)

    def read(self):
        ret, image = self.video_capture.read()
        if ret:
            if USE_CALIBRATION:
                image = self.camera_calibration.undistort(image)
        return ret, image


def main():
    camera_calibration = CameraCalibration(CAMERA_ID, True)
    camera_calibration.calibrate()
    # TODO
    # camera_calibration.save_calibration()
    # camera_calibration.load_calibration()
    cap = UndistortedVideoCapture(CAMERA_ID, camera_calibration)
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
