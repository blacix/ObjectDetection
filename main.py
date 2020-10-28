# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.

# https://tensorflow-object-detection-api-tutorial.readthedocs.io/en/latest/install.html#install-the-tensorflow-pip-package

import tensorflow as tf
import cv2 as cv
import time
from object_detection.utils import label_map_util
import numpy as np
from PIL import Image
import time
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as viz_utils

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import os
import pathlib

category_index = ""
IMAGE_PATHS = ""


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




def load_model():
    # %%
    # Load the model
    # ~~~~~~~~~~~~~~
    # Next we load the downloaded model


    # PATH_TO_SAVED_MODEL = PATH_TO_MODEL_DIR + "/saved_model"
    # PATH_TO_SAVED_MODEL = "workspace/training_demo/pre-trained-models/ssd_mobilenet_v2_320x320_coco17_tpu-8/saved_model"
    # PATH_TO_SAVED_MODEL = "workspace/training_demo/pre-trained-models/ssd_resnet50_v1_fpn_640x640_coco17_tpu-8/saved_model"

    PATH_TO_SAVED_MODEL = "workspace/training_demo/exported-models/my_model/saved_model"

    print('Loading model...', end='')
    start_time = time.time()

    # Load saved model and build the detection function
    detect_fn = tf.saved_model.load(PATH_TO_SAVED_MODEL)

    end_time = time.time()
    elapsed_time = end_time - start_time
    print('Done! Took {} seconds'.format(elapsed_time))

    return detect_fn


def load_label_map():
    # PATH_TO_LABELS = "models/research/object_detection/data/mscoco_label_map.pbtxt"
    PATH_TO_LABELS = "workspace/training_demo/annotations/label_map.pbtxt"

    # %%
    # Load label map data (for plotting)
    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    # Label maps correspond index numbers to category names, so that when our convolution network
    # predicts `5`, we know that this corresponds to `airplane`.  Here we use internal utility
    # functions, but anything that returns a dictionary mapping integers to appropriate string labels
    # would be fine.

    return label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS,
                                                                        use_display_name=True)
def download_images():
    base_url = 'https://raw.githubusercontent.com/tensorflow/models/master/research/object_detection/test_images/'
    filenames = ['image1.jpg', 'image2.jpg']
    image_paths = []
    for filename in filenames:
        image_path = tf.keras.utils.get_file(fname=filename,
                                            origin=base_url + filename,
                                            untar=False)
        image_path = pathlib.Path(image_path)
        image_paths.append(str(image_path))
    return image_paths


def doit():
    IMAGE_PATHS = download_images()
    # IMAGE_PATHS = ["001.png"]
    cap = cv.VideoCapture(1)
    while(True):

        ret, image_np = cap.read()

    # for image_path in IMAGE_PATHS:
    #     print('Running inference for {}... '.format(image_path), end='')

        # image_np = load_image_into_numpy_array(image_path)

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
        detections = detect_fn(input_tensor)

        # All outputs are batches tensors.
        # Convert to numpy arrays, and take index [0] to remove the batch dimension.
        # We're only interested in the first num_detections.
        num_detections = int(detections.pop('num_detections'))
        detections = {key: value[0, :num_detections].numpy()
                      for key, value in detections.items()}
        detections['num_detections'] = num_detections

        # print(detections)

        # detection_classes should be ints.
        detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

        image_np_with_detections = image_np.copy()

        viz_utils.visualize_boxes_and_labels_on_image_array(
            image_np_with_detections,
            detections['detection_boxes'],
            detections['detection_classes'],
            detections['detection_scores'],
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=100,
            min_score_thresh=.50,
            agnostic_mode=False)

        cv.imshow("yeee", image_np_with_detections)
        cv.waitKey(1)


if __name__ == '__main__':
    detect_fn = load_model()
    category_index = load_label_map()
    doit()

