import os

CAMERA_ID = 1
USE_CALIBRATION = False
USE_GPU = True
MIN_SCORE_THRESHOLD = .60

# MODEL_NAME = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
MODEL_NAME = "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"
# MODEL_NAME = "ssd_mobilenet_v2_fpnlite_640x640_coco17_tpu-8"

TF_MODELS_REPOSITORY_DIR = "models-master"
TRAINING_DIR = "model-training"
MODEL_DIR = os.path.join(TRAINING_DIR, "models", MODEL_NAME)
PRE_TRAINED_MODEL_DIR = os.path.join(TRAINING_DIR, "pre-trained-models", MODEL_NAME)
EXPORT_DIR = os.path.join(TRAINING_DIR, "exported-models", MODEL_NAME)

ANNOTATIONS_DIR = os.path.join(TRAINING_DIR, "annotations")
LABEL_FILE = os.path.join(ANNOTATIONS_DIR, "label_map.pbtxt")
TEST_RECORD_FILE = os.path.join(ANNOTATIONS_DIR, "test.record")
TRAIN_RECORD_FILE = os.path.join(ANNOTATIONS_DIR, "train.record")

# for running
PATH_TO_SAVED_MODEL = os.path.join(EXPORT_DIR, "saved_model")
# PATH_TO_SAVED_MODEL = os.path.join(PRE_TRAINED_MODEL_DIR, "saved_model")
PATH_TO_LABELS = LABEL_FILE
# PATH_TO_LABELS = os.path.join(TF_MODELS_REPOSITORY_DIR, "research", "object_detection", "data", "mscoco_label_map.pbtxt")

PIPELINE_CONFIG = os.path.join(MODEL_DIR, "pipeline.config")

OBJ_DETECTION_SCRIPTS_DIR = os.path.join(TF_MODELS_REPOSITORY_DIR, "research", "object_detection")
TRAIN_SCRIPT = os.path.join(OBJ_DETECTION_SCRIPTS_DIR, "model_main_tf2.py")
EXPORT_SCRIPT = os.path.join(OBJ_DETECTION_SCRIPTS_DIR, "exporter_main_v2.py")

IMAGES_DIR = os.path.join(TRAINING_DIR, "images")
TEST_IMAGES_DIR = os.path.join(IMAGES_DIR, "test")
TRAIN_IMAGES_DIR = os.path.join(IMAGES_DIR, "train")
