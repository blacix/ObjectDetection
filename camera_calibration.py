#!/usr/bin/env python

# Python 2/3 compatibility
from __future__ import print_function

import numpy as np
import cv2 as cv
import glob
from config import CAMERA_ID
# https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-333b05afa0b0
# https://medium.com/@kennethjiang/calibrate-fisheye-lens-using-opencv-part-2-13990f1b157f


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
            self.pattern_points = np.zeros((1, self.checkboard_pattern_size[0] * self.checkboard_pattern_size[1], 3),
                                           np.float32)
            self.pattern_points[0, :, :2] = np.mgrid[0:self.checkboard_pattern_size[0],
                                            0:self.checkboard_pattern_size[1]].T.reshape(-1, 2)
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

        self.camera_matrix = np.zeros((3, 3))
        self.dist_coeffs = np.zeros((4, 1))

        rvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]
        tvecs = [np.zeros((1, 1, 3), dtype=np.float64) for i in range(N_OK)]

        print('calibrating fisheye camera...')

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

        print("calculating optimal camera matrix...")
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
            # criteria and flags were not set
            ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv.calibrateCamera(self.obj_points, self.img_points, (w, h),
                                                                               self.fisheye_calibration_flags,
                                                                               self.fisheye_calibration_criteria)

            print("calculating optimal camera matrix...")
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
            return self.undistort_image(image)

    def undistort_fisheye_image(self, image):
        # with remap
        # map1, map2 = cv.fisheye.initUndistortRectifyMap(self.scaled_K, self.dist_coeffs, np.eye(3),
        #                                                 self.new_camera_matrix, self.dim3, cv.CV_16SC2)
        # undistorted_img = cv.remap(image, map1, map2, interpolation=cv.INTER_LINEAR, borderMode=cv.BORDER_CONSTANT)

        undistorted_img = cv.fisheye.undistortImage(image, self.camera_matrix, self.dist_coeffs, None,
                                                    self.new_camera_matrix)
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

    def save_calibration(self, name=None):
        print('Saving calibration...')
        if name is None:
            file_name = 'calibration.yml'
        else:
            file_name = f'calibration_{name}.yml'
        print(f'Saving calibration {file_name}...')
        # writes array to .yml file
        fs_write = cv.FileStorage(file_name, cv.FILE_STORAGE_WRITE)
        arr = np.random.rand(5, 5)
        fs_write.write("camera_matrix", self.camera_matrix)
        fs_write.write("new_camera_matrix", self.new_camera_matrix)
        fs_write.write("dist_coeffs", self.dist_coeffs)
        fs_write.release()

    def load_calibration(self, name: str = None):
        if name is None:
            file_name = 'calibration.yml'
        else:
            file_name = f'calibration_{name}.yml'

        print(f'Loading calibration {file_name}...')
        fs_read = cv.FileStorage(file_name, cv.FILE_STORAGE_READ)
        self.camera_matrix = fs_read.getNode('camera_matrix').mat()
        self.new_camera_matrix = fs_read.getNode('new_camera_matrix').mat()
        self.dist_coeffs = fs_read.getNode('dist_coeffs').mat()
        # TODO
        # self.fisheye = fs_read.getNode('fisheye')
        fs_read.release()

    def is_camera_calibrated(self):
        return self.camera_matrix is not None and self.new_camera_matrix is not None and self.dist_coeffs is not None


class UndistortedVideoCapture:
    def __init__(self, camera_id, fisheye=False, camera_calibration=None):
        if camera_calibration is None:
            self.camera_calibration = CameraCalibration(camera_id, fisheye)
            # self.camera_calibration.load_calibration()
        else:
            self.camera_calibration = camera_calibration
        self.video_capture = cv.VideoCapture(camera_id)
        # https://raspberrypi.stackexchange.com/questions/105358/raspberry-pi4-error-while-using-2-usb-cameras-vidioc-qbuf-invalid-argument/
        # set to compressed format, so USB won't hang
        self.video_capture.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc('M', 'J', 'P', 'G'))
        # self.video_capture.set(cv.CAP_PROP_FRAME_WIDTH, 1024)
        # self.video_capture.set(cv.CAP_PROP_FRAME_HEIGHT, 768)

    def read(self):
        ret, image = self.video_capture.read()
        if ret:
            image = self.camera_calibration.undistort(image)
        return ret, image


def main():
    camera_calibration = CameraCalibration(CAMERA_ID, fisheye=True)
    camera_calibration.calibrate()
    camera_calibration.save_calibration()
    cap = UndistortedVideoCapture(CAMERA_ID, fisheye=True)
    cap.camera_calibration.load_calibration()
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
