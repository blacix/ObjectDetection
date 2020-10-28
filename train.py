import os
from config import *


def generate_tf_records():
    print("generating TF records...")
    command = "python generate_tfrecord.py -x " + TRAIN_IMAGES_DIR + " -l " + LABEL_FILE + " -o " + TRAIN_RECORD_FILE
    os.system(command)
    command = "python generate_tfrecord.py -x " + TEST_IMAGES_DIR + " -l " + LABEL_FILE + " -o " + TEST_RECORD_FILE
    os.system(command)


def train_model():
    print("training model {}".format(MODEL_NAME))
    command = "python " + TRAIN_SCRIPT + " --model_dir=" + MODEL_DIR + " --pipeline_config_path=" + PIPELINE_CONFIG
    print("command: {}".format(command))
    if not USE_GPU:
        os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

    return os.system(command)
    # command += " --checkpoint_dir=" + MODEL_DIR
    # os.system(command)


def export_model():
    print("exporting model {}".format(MODEL_NAME))
    command = "python " + EXPORT_SCRIPT + " --input_type image_tensor --pipeline_config_path " + PIPELINE_CONFIG + " --trained_checkpoint_dir " + MODEL_DIR + " --output_directory " + EXPORT_DIR
    print("command: {}".format(command))
    os.system(command)


if __name__ == '__main__':
    generate_tf_records()
    result = train_model()
    export_model()
