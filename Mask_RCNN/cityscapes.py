#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
sys.path.append("Mask_RCNN")
import random
import math
import re
import time
import numpy as np
import cv2
import skimage
import matplotlib
import matplotlib.pyplot as plt

from mrcnn.config import Config
from mrcnn import utils
import mrcnn.model as modellib
from mrcnn import visualize
from mrcnn.model import log

# Root directory of the project
ROOT_DIR = os.getcwd()

# Directory to save logs and trained model
MODEL_DIR = os.path.join(ROOT_DIR, "logs")

# Directory to load source dataset
DATA_DIR = os.path.join(ROOT_DIR, "../data/leftImg8bit")

# Directory to load groundtruth dataset
MASK_DIR = os.path.join(ROOT_DIR, "../data/gtFine")

# Local path to trained weights file
COCO_MODEL_PATH = os.path.join(ROOT_DIR, "mask_rcnn_coco.h5")

# Download COCO trained weights from Releases if needed
if not os.path.exists(COCO_MODEL_PATH):
    utils.download_trained_weights(COCO_MODEL_PATH)

TRAINING = True
subset = ''


class CityscapeConfig(Config):
    """Configuration for training on the cityscape dataset.
    Derives from the base Config class and overrides values specific
    to the cityscape dataset.
    """
    # Give the configuration a recognizable name
    NAME = "cityscape"

    # We use a GPU with 12GB memory.
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1  # 8

    # Number of training steps per epoch
    STEPS_PER_EPOCH = 6000

    # Number of validation steRPNps to run at the end of every training epoch.
    VALIDATION_STEPS = 300

    # Backbone network architecture
    # Supported values are: resnet50, resnet101
    BACKBONE = "resnet50"

    # Number of classes (including background)
    NUM_CLASSES = 1 + 6  # background + 1 shapes

    # Input image resing
    IMAGE_RESIZE_MODE = "square"
    IMAGE_MIN_DIM = 800
    IMAGE_MAX_DIM = 1024

    # Learning rate and momentum
    LEARNING_RATE = 0.01


config = CityscapeConfig()
# config.display()


def get_ax(rows=1, cols=1, size=8):
    """Return a Matplotlib Axes array to be used in
    all visualizations in the notebook. Provide a
    central point to control graph sizes.
    Change the default size attribute to control the size
    of rendered images
    """
    _, ax = plt.subplots(rows, cols, figsize=(size * cols, size * rows))
    return ax


class CityscapeDataset(utils.Dataset):
    '''
    load_shapes()
    load_image()
    load_mask()
    '''

    def __init__(self, subset):
        super(CityscapeDataset, self).__init__(self)
        self.subset = subset

    def load_shapes(self):
        """
        subset: "train"/"val"
        image_id: use index to distinguish the images.
        gt_id: ground truth(mask) id.
        height, width: the size of the images.
        path: directory to load the images.
        """
        # Add classes you want to train
        self.add_class("cityscape", 1, "car")
        self.add_class("cityscape", 2, "truck")
        self.add_class("cityscape", 3, "bus")
        self.add_class("cityscape", 4, "train")
        self.add_class("cityscape", 5, "motorcycle")
        self.add_class("cityscape", 6, "bicycle")

        # Add images
        image_dir = "{}/{}".format(DATA_DIR, self.subset)
        image_ids = os.listdir(image_dir)
        for index, item in enumerate(image_ids):
            temp_image_path = "{}/{}".format(image_dir, item)
            temp_image_size = skimage.io.imread(temp_image_path).shape
            self.add_image("cityscape", image_id=index, gt_id=os.path.splitext(item)[0],
                            height=temp_image_size[0], width=temp_image_size[1],
                            path=temp_image_path)

    def load_image(self, image_id):
        """Load images according to the given image ID."""
        info = self.image_info[image_id]
        image = skimage.io.imread(info['path'])
        return image

    def image_reference(self, image_id):
        """Return the shapes data of the image."""
        info = self.image_info[image_id]
        if info["source"] == "cityscape":
            return info["cityscape"]
        else:
            super(self.__class__).image_reference(self, image_id)

    def load_mask(self, image_id):
        """Load instance masks of the given image ID.
        count: the number of masks in each image.
        class_id: the first letter of each mask file's name.
        """
        info = self.image_info[image_id]
        gt_id = info['gt_id']
        mask_dir = "{}/{}/{}".format(MASK_DIR, self.subset, gt_id)
        masks_list = os.listdir(mask_dir)
        count = len(masks_list)
        mask = np.zeros([info['height'], info['width'], count])
        class_ids = []

        for index, item in enumerate(masks_list):
            temp_mask_path = "{}/{}".format(mask_dir, item)
            tmp_mask = 255 - skimage.io.imread(temp_mask_path)[:, :, np.newaxis]
            mask[:, :, index:index+1] = tmp_mask
            class_ids.append(item[0])

        # Handle occlusions
        occlusion = np.logical_not(mask[:, :, -1]).astype(np.uint8)
        for i in range(count-2, -1, -1):
            mask[:, :, i] = mask[:, :, i] * occlusion
            occlusion = np.logical_and(occlusion, np.logical_not(mask[:, :, i]))
        return mask, np.array(class_ids, dtype=np.uint8)

if TRAINING:
    # Training dataset
    dataset_train = CityscapeDataset("train")
    dataset_train.load_shapes()
    dataset_train.prepare()

# Validation dataset
dataset_val = CityscapeDataset("val")
dataset_val.load_shapes()
dataset_val.prepare()

# Inspect the dataset
# print("Image Count: {}".format(len(dataset_train.image_ids)))
# print("Class Count: {}".format(dataset_train.num_classes))
# for i, info in enumerate(dataset_train.class_info):
#     print("{:3}. {:50}".format(i, info['name']))

# Load and display random samples
# image_ids = np.random.choice(dataset_train.image_ids, 3)
# for image_id in image_ids:
#     image = dataset_train.load_image(image_id)
#     mask, class_ids = dataset_train.load_mask(image_id)
#     visualize.display_top_masks(image, mask, class_ids,
#                                 dataset_train.class_names)


# Create model in training mode
model = modellib.MaskRCNN(mode="training", config=config,
                          model_dir=MODEL_DIR)

# Which weights to start with?
init_with = "coco"  # imagenet, coco, or last

if init_with == "imagenet":
    model.load_weights(model.get_imagenet_weights(), by_name=True)
elif init_with == "coco":
    # Load weights trained on MS COCO, but skip layers that
    # are different due to the different number of classes
    # See README for instructions to download the COCO weights
    model.load_weights(COCO_MODEL_PATH, by_name=True,
                       exclude=["mrcnn_class_logits", "mrcnn_bbox_fc",
                                "mrcnn_bbox", "mrcnn_mask"])
elif init_with == "last":
    # Load the last model you trained and continue training
    model.load_weights(model.find_last()[1], by_name=True)

if TRAINING:
    # Train the head branches
    # Passing layers="heads" freezes all layers except the head
    # layers. You can also pass a regular expression to select
    # which layers to train by name pattern.
    # model.train(dataset_train, dataset_val,
    #             learning_rate=config.LEARNING_RATE,
    #             epochs=1,
    #             layers='heads')

    # Fine tune all layers
    # Passing layers="all" trains all layers. You can also
    # pass a regular expression to select which layers to
    # train by name pattern.
    # learning_rate = 0.01
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE,
                epochs=3,
                layers="all")

    # learning_rate = 0.001
    model.train(dataset_train, dataset_val,
                learning_rate=config.LEARNING_RATE / 10,
                epochs=4,
                layers="all")

    # Save weights
    # Typically not needed because callbacks save after every epoch
    # Uncomment to save manually
    # model_path = os.path.join(MODEL_DIR, "mask_rcnn_shapes.h5")
    # model.keras_model.save_weights(model_path)


class InferenceConfig(CityscapeConfig):
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    IMAGE_MIN_DIM = 1024
    IMAGE_MAX_DIM = 1024

inference_config = InferenceConfig()

# Recreate the model in inference mode
model = modellib.MaskRCNN(mode="inference",
                          config=inference_config,
                          model_dir=MODEL_DIR)

# Get path to saved weights
# Either set a specific path or find last trained weights
# model_path = os.path.join(ROOT_DIR, ".h5 file name here")
model_path = model.find_last()[1]

# Load trained weights (fill in path to trained weights here)
assert model_path != "", "Provide path to trained weights"
print("Loading weights from ", model_path)

model.load_weights(model_path, by_name=True)

# Test on random images
image_ids = np.random.choice(dataset_val.image_ids, 2)
for image_id in image_ids:
    original_image, image_meta, gt_class_id, gt_bbox, gt_mask =\
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)

	# log("original_image", original_image)
	# log("image_meta", image_meta)
	# log("gt_class_id", gt_class_id)
	# log("gt_bbox", gt_bbox)
	# log("gt_mask", gt_mask)

    # visualize.display_instances(original_image, gt_bbox, gt_mask, gt_class_id,
    #                         dataset_val.class_names, figsize=(8, 8))

    results = model.detect([original_image], verbose=1)
    r = results[0]
    visualize.display_instances(original_image, r['rois'], r['masks'], r['class_ids'],
                                dataset_val.class_names, r['scores'], ax=get_ax())


# Compute mAP
#APs_05: IoU = 0.5
#APs_all: IoU from 0.5-0.95 with increments of 0.05
image_ids = np.random.choice(dataset_val.image_ids, 489)
APs_05 = []
APs_all = []

for image_id in image_ids:
    # Load images and ground truth data
    image, image_meta, gt_class_id, gt_bbox, gt_mask = \
        modellib.load_image_gt(dataset_val, inference_config,
                               image_id, use_mini_mask=False)
    molded_images = np.expand_dims(modellib.mold_image(image, inference_config), 0)
    # Run object detection
    results = model.detect([image], verbose=0)
    r = results[0]
    # Compute AP
    AP_05, precisions, recalls, overlaps = \
        utils.compute_ap(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs_05.append(AP_05)

    AP_all = \
        utils.compute_ap_range(gt_bbox, gt_class_id, gt_mask,
                         r["rois"], r["class_ids"], r["scores"], r['masks'])
    APs_all.append(AP_all)

print("mAP: ", np.mean(APs_05))
print("mAP: ", np.mean(APs_all))






