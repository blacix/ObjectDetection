MODEL_NAME = "ssd_mobilenet_v2_320x320_coco17_tpu-8"
# MODEL_NAME = "ssd_resnet50_v1_fpn_640x640_coco17_tpu-8"

USE_GPU = True

TF_MODELS_REPOSITORY_DIR = "tf_models_repository/"
TRAINING_DIR = "workspace/training_demo/"
MODEL_DIR = TRAINING_DIR + "models/" + MODEL_NAME + "/"
PRE_TRAINED_MODEL_DIR = TRAINING_DIR + "pre-trained-models/" + MODEL_NAME + "/"
ANNOTATIONS_DIR = TRAINING_DIR + "annotations/"
EXPORT_DIR = TRAINING_DIR + "exported-models/" + MODEL_NAME + "/"

LABEL_FILE = ANNOTATIONS_DIR + "label_map.pbtxt"

PATH_TO_SAVED_MODEL = EXPORT_DIR + "saved_model/"
# PATH_TO_SAVED_MODEL = PRE_TRAINED_MODEL_DIR + "saved_model/"

PATH_TO_LABELS = LABEL_FILE
# PATH_TO_LABELS = TF_MODELS_REPOSITORY_DIR + "/research/object_detection/data/mscoco_label_map.pbtxt"


PRE_TRAINED_CHECKPOINT = PRE_TRAINED_MODEL_DIR + "checkpoint/"
PIPELINE_CONFIG = MODEL_DIR + "pipeline.config"

OBJ_DETECTION_SCRIPTS_DIR = TF_MODELS_REPOSITORY_DIR + "research/object_detection/"
TRAIN_SCRIPT = OBJ_DETECTION_SCRIPTS_DIR + "model_main_tf2.py"
EXPORT_SCRIPT = OBJ_DETECTION_SCRIPTS_DIR + "exporter_main_v2.py"

TEST_RECORD_FILE = ANNOTATIONS_DIR + "test.record"
TRAIN_RECORD_FILE = ANNOTATIONS_DIR + "train.record"

IMAGES_DIR = "workspace/training_demo/images/"
TEST_IMAGES_DIR = IMAGES_DIR + "test/"
TRAIN_IMAGES_DIR = IMAGES_DIR + "train/"


