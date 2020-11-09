import os
import json
import argparse
import numpy as np
import random
import math
from PIL import Image, ImageEnhance
from pathlib import Path

import xml.etree.cElementTree as ET
import cv2 as cv

# Entrypoint Args
parser = argparse.ArgumentParser(description='Create synthetic training data for object detection algorithms.')
parser.add_argument("-bkg", "--backgrounds", type=str, default="Backgrounds/",
                    help="Path to background images folder.")
parser.add_argument("-obj", "--objects", type=str, default="Objects/",
                    help="Path to object images folder.")
parser.add_argument("-o", "--output", type=str, default="TrainingImages/",
                    help="Path to output images folder.")
parser.add_argument("-ann", "--annotate", type=bool, default=True,
                    help="Include annotations in the data augmentation steps?")
parser.add_argument("-s", "--sframe", type=bool, default=False,
                    help="Convert dataset to an sframe?")
parser.add_argument("-g", "--groups", type=bool, default=False,
                    help="Include groups of objects in training set?")
parser.add_argument("-mut", "--mutate", type=bool, default=True,
                    help="Perform mutatuons to objects (rotation, brightness, shapness, contrast)")
args = parser.parse_args()


# Prepare data creation pipeline
base_bkgs_path = args.backgrounds
bkg_images = [f for f in os.listdir(base_bkgs_path) if not f.startswith(".")]
objs_path = args.objects
obj_images = [f for f in os.listdir(objs_path) if not f.startswith(".")]
sizes = [0.4, 0.6, 0.8, 1, 1.2] # different obj sizes to use TODO make configurable
count_per_size = 4 # number of locations for each obj size TODO make configurable
annotations = [] # store annots here
output_images = args.output
n = 1


# Helper functions
def get_obj_positions(obj, bkg, count=1):
    obj_w, obj_h = [], []
    x_positions, y_positions = [], []
    bkg_w, bkg_h = bkg.size
    # Rescale our obj to have a couple different sizes
    obj_sizes = [tuple([int(s*x) for x in obj.size]) for s in sizes]
    for w, h in obj_sizes:
        obj_w.extend([w]*count)
        obj_h.extend([h]*count)
        # TODO
        max_x, max_y = bkg_w-w, bkg_h-h
        x_positions.extend(list(np.random.randint(0, max_x, count)))
        y_positions.extend(list(np.random.randint(0, max_y, count)))
    return obj_h, obj_w, x_positions, y_positions


def mutate_image(img):
    # resize image for random value
    resize_rate = random.choice(sizes)
    img = img.resize([int(img.width*resize_rate), int(img.height*resize_rate)], Image.BILINEAR)

    # rotate image for random andle and generate exclusion mask 
    rotate_angle = random.randint(0, 360)
    mask = Image.new('L', img.size, 255)
    sx1, sy1 = img.size
    img = img.rotate(rotate_angle, expand=True)
    sx2, sy2 = img.size
    mask = mask.rotate(rotate_angle, expand=True)
    # https://stackoverflow.com/questions/7501009/affine-transform-in-pil-python
    # Image.transform(size, Image.AFFINE, data=None, resample=0, fill=1)

    # perform some enhancements on image
    # enhancers = [ImageEnhance.Brightness, ImageEnhance.Color, ImageEnhance.Contrast, ImageEnhance.Sharpness]
    enhancers = [ImageEnhance.Brightness, ImageEnhance.Sharpness]
    enhancers_count = random.randint(0, len(enhancers))
    for i in range(0, enhancers_count):
        enhancer = random.choice(enhancers)
        enhancers.remove(enhancer)
        img = enhancer(img).enhance(random.uniform(0.5, 1.5))

    return img, mask


def scale_rotate_translate(image, angle, center=None, new_center=None, scale=None, expand=False):
    if center is None:
        return image.rotate(angle, expand=expand)

    angle = 30
    angle = -angle/180.0*math.pi

    nx, ny = x, y = center
    sx=sy=1.0
    if new_center:
        (nx, ny) = new_center
    if scale:
        (sx, sy) = scale
    cosine = math.cos(angle)
    sine = math.sin(angle)

    a = cosine/sx
    b = sine/sx + math.tan(angle)
    c = x-nx*a-ny*b
    d = -sine/sy
    e = cosine/sy
    f = y-nx*d-ny*e

    # a = cosine/sx
    # b = sine/sx
    # c = x-nx*a-ny*b
    # d = -sine/sy
    # e = cosine/sy
    # f = y-nx*d-ny*e
    return image.transform(image.size, Image.AFFINE, (a, b, c, d, e, f), resample=Image.BICUBIC)


def create_xml(object_label, out_filename, xmin, ymin, xmax, ymax, img_width, img_height, truncated):
    tag_annotation = ET.Element("annotation")
    tag_folder = ET.SubElement(tag_annotation, "folder").text = "./"
    tag_image_file_name = ET.SubElement(tag_annotation, "filename").text = "{}.png".format(out_filename)
    tag_path = ET.SubElement(tag_annotation, "path").text = "{}.png".format(out_filename)

    tag_source = ET.SubElement(tag_annotation, "source")
    tag_database = ET.SubElement(tag_source, "source").text = "Unknown"

    tag_size = ET.SubElement(tag_annotation, "size")
    width = ET.SubElement(tag_size, "width").text = str(img_width)
    height = ET.SubElement(tag_size, "height").text = str(img_height)
    depth = ET.SubElement(tag_size, "depth").text = "3"

    segmented = ET.SubElement(tag_annotation, "segmented").text = "0"

    tag_object = ET.SubElement(tag_annotation, "object")
    tag_name = ET.SubElement(tag_object, "name").text = object_label
    tag_pose = ET.SubElement(tag_object, "pose").text = "Unspecified"
    tag_truncated = ET.SubElement(tag_object, "truncated").text = "{}".format("1" if truncated else "0")
    tag_difficult = ET.SubElement(tag_object, "difficult").text = "0"

    tag_bndbox = ET.SubElement(tag_object, "bndbox")
    tag_xmin = ET.SubElement(tag_bndbox, "xmin").text = str(xmin)
    tag_ymin = ET.SubElement(tag_bndbox, "ymin").text = str(ymin)
    tag_xmax = ET.SubElement(tag_bndbox, "xmax").text = str(xmax)
    tag_ymax = ET.SubElement(tag_bndbox, "ymax").text = str(ymax)
    #
    # ET.SubElement(doc, "field1", name="blah").text = "some value1"
    # ET.SubElement(doc, "field2", name="asdfasd").text = "some vlaue2"

    tree = ET.ElementTree(tag_annotation)
    tree.write("TrainingImages/{}.xml".format(out_filename))

#
# def get_box(obj_w, obj_h, max_x, max_y):
#     x1, y1 = np.random.randint(0, max_x, 1), np.random.randint(0, max_y, 1)
#     x2, y2 = x1 + obj_w, y1 + obj_h
#     return [x1[0], y1[0], x2[0], y2[0]]
#
#
# # check if two boxes intersect
# def intersects(box, new_box):
#     box_x1, box_y1, box_x2, box_y2 = box
#     x1, y1, x2, y2 = new_box
#     return not (box_x2 < x1 or box_x1 > x2 or box_y1 > y2 or box_y2 < y1)


def resize_images(images):
    cnt = 1
    for i in images:
        image_path = base_bkgs_path + i
        image = Image.open(image_path)
        resized = image.resize(size=(800, 800))
        fp = output_images + "background" + str(cnt) + ".png"
        # Save the image
        resized.save(fp=fp, format="png")
        cnt = cnt + 1


if __name__ == "__main__":

    # Make synthetic training data
    print("Making synthetic images.", flush=True)
    # resize_images(bkg_images)
    # exit()
    # for bkg in bkg_images:
    bkg = bkg_images[0]
    if True:
        # Load the background image
        bkg_path = base_bkgs_path + bkg
        bkg_img = Image.open(bkg_path)
        bkg_x, bkg_y = bkg_img.size

        for i in obj_images:
            # Load the single obj
            i_path = objs_path + i
            obj_img = Image.open(i_path)

            # Get an array of random obj positions (from top-left corner)
            obj_h, obj_w, x_pos, y_pos = get_obj_positions(obj=obj_img, bkg=bkg_img, count=count_per_size)

            # Create synthetic images based on positions
            for h, w, x, y in zip(obj_h, obj_w, x_pos, y_pos):
                # Copy background
                bkg_w_obj = bkg_img.copy()

                # new_obj = scale_rotate_translate(obj_img, 90, center=(w//2, h//2), new_center=None, scale=(1.0, 1.0), expand=True)

                new_obj, mask = mutate_image(obj_img)

                bkg_width, bkg_height = bkg_img.size
                print("n {} bkg_width {} bkg_height {}".format(n, bkg_width, bkg_height))

                # size of the mutated image == bounding box
                obj_img_width, obj_img_height = new_obj.size
                xmax = x + obj_img_width
                ymax = y + obj_img_height
                xmin = x - obj_img_width
                ymin = y - obj_img_height
                truncated = xmax >= bkg_width or ymax >= bkg_height or xmin <= 0 or ymin <= 0
                print("n {}  xmax {} ymax {} xmin {} ymin {} ymax {} truncated {}".format(n, xmax, ymax, xmin, ymin,
                                                                                          ymax, truncated))
                if truncated:
                    print("skipping {}".format(n))
                    continue

                print("n {}  w {} h {} ".format(n, w, h))
                print("n {} height {} width {} x {} y {}".format(n, h, w, x + (0.5 * w), y + (0.5 * h)))


                create_xml(Path(i).stem, # i.split(".png")[0]
                           "{}".format(n), x, y, xmax, ymax, bkg_width, bkg_height, truncated)

                # Paste on the obj
                # new_obj = obj_img.resize(size=(w, h))
                # bkg_w_obj.paste(new_obj, (x, y), mask)
                bkg_w_obj.paste(new_obj, (x, y))

                output_fp = output_images + str(n) + ".png"
                # Save the image
                bkg_w_obj.save(fp=output_fp, format="png")
                cv_image = cv.imread(output_images + str(n) + ".png")
                cv.imshow("asd", cv_image)

                n += 1

    total_images = len([f for f in os.listdir(output_images) if not f.startswith(".")])
    print("Done! Created {} synthetic training images.".format(total_images), flush=True)
    cv.waitKey(5000)
