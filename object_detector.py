# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#install-the-tensorflow-pip-package

import time
import tensorflow as tf
import cv2 as cv
import numpy as np
from PIL import Image

from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import config
import camera_calibration as calib


class ObjectDetector:
    def __init__(self):
        self.detection_function = self.load_model()
        self.category_index = self.load_label_map()

    @staticmethod
    def load_image_into_numpy_array(path):
        """Load an image from file into a numpy array.

        Puts image into numpy array to feed into tensorflow graph.
        Note that by convention we put it into a numpy array with shape
        (height, width, channels), where channels=3 for RGB.

        Args:
          path: the file path to the image

        Returns:
          uint8 numpy array with shape (img_height, img_width, 3)
        """
        return np.array(Image.open(path))

    @staticmethod
    def load_model():
        print('Loading model...')
        start_time = time.time()

        # Load saved model and build the detection function
        detection_function = tf.saved_model.load(config.PATH_TO_SAVED_MODEL)

        end_time = time.time()
        elapsed_time = end_time - start_time
        print('Done! Took {} seconds'.format(elapsed_time))
        return detection_function

    @staticmethod
    def load_label_map():
        return label_map_util.create_category_index_from_labelmap(config.PATH_TO_LABELS, use_display_name=True)

    def detect_objects(self, image):
        image_np = np.array(image)
        # image_np = np.array(image)

        # Things to try:
        # Flip horizontally
        # image_np = np.fliplr(image_np).copy()

        # Convert image to grayscale
        # image_np = np.tile(
        #     np.mean(image_np, 2, keepdims=True), (1, 1, 3)).astype(np.uint8)

        # The input needs to be a tensor, convert it using `tf.convert_to_tensor`.
        input_tensor = tf.convert_to_tensor(image_np)
        # The model expects a batch of images, so add an axis with `tf.newaxis`.
        input_tensor = input_tensor[tf.newaxis, ...]

        # input_tensor = np.expand_dims(image_np, 0)
        detections = self.detection_function(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            self.category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=config.MIN_SCORE_THRESHOLD,
            agnostic_mode=False)

        return image_np_with_detections


if __name__ == '__main__':
    cap = calib.UndistortedVideoCapture(config.CAMERA_ID)
    object_detector = ObjectDetector()

    while True:
        ret, image = cap.read()
        if not ret:
            continue

        image = object_detector.detect_objects(image)
        cv.imshow('object detection', image)
        if cv.waitKey(100) == ord('q'):
            break

    cv.destroyAllWindows()
