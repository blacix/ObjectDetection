import os
from config import *


def generate_tf_records():
    command = "python generate_tfrecord.py -x " + TRAIN_IMAGES_DIR + " -l " + LABEL_FILE + " -o " + TRAIN_RECORD_FILE
    os.system(command)
    command = "python generate_tfrecord.py -x " + TEST_IMAGES_DIR + " -l " + LABEL_FILE + " -o " + TEST_RECORD_FILE
    os.system(command)


def train_model():
    # python model_main_tf2.py --model_dir=models/ssd_mobilenet_v2_320x320_coco17_tpu-8 --pipeline_config_path=models/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config
    # check model
    # python model_main_tf2.py --model_dir=models/ssd_mobilenet_v2_320x320_coco17_tpu-8 --pipeline_config_path=models/ssd_mobilenet_v2_320x320_coco17_tpu-8/pipeline.config --checkpoint_dir=models/ssd_mobilenet_v2_320x320_coco17_tpu-8
    command = "python " + TRAIN_SCRIPT + " --model_dir=" + MODEL_DIR + " --pipeline_config_path=" + PIPELINE_CONFIG
    print(command + "\n")
    # os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
    return os.system(command)
    # command += " --checkpoint_dir=" + MODEL_DIR
    # os.system(command)


def export_model():
    # "python .\exporter_main_v2.py --input_type image_tensor --pipeline_config_path .\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\pipeline.config --trained_checkpoint_dir .\models\ssd_mobilenet_v2_320x320_coco17_tpu-8\ --output_directory .\exported-models\my_model"
    command =  "python " + EXPORT_SCRIPT + " --input_type image_tensor --pipeline_config_path " + PIPELINE_CONFIG + " --trained_checkpoint_dir " + MODEL_DIR + " --output_directory " + EXPORT_DIR
    os.system(command)


if __name__ == '__main__':
    # generate_tf_records()
    # result = train_model()
    export_model()
