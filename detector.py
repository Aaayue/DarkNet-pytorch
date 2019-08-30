from __future__ import division
import os
import torch
import torch.nn as nn
import numpy as np
import cv2
import argparse
from util import *
from darknet import DarkNet
import time
import pandas as pd
import pickle as pkl
import random
import math


def add_parse():

    parse = argparse.ArgumentParser(description="YOLO-v3 object detection")
    parse.add_argument(
        "--images",
        dest="imgs",
        help="Name/Directory of image files"
    )
    parse.add_argument(
        "--outdet",
        dest="det",
        help="Direstory to save detect results"
    )
    parse.add_argument(
        "--cfg",
        dest="cfg",
        help="Configure file path"
    )
    parse.add_argument(
        "--weights",
        dest="weights",
        help="Weights file path"
    )
    parse.add_argument(
        "--batch",
        dest="bs",
        help="Batch size",
        default=2,
        type=int
    )
    parse.add_argument(
        "--confidence",
        dest="conf",
        help="Threshold of objectness score",
        type=float,
        default=0.5
    )
    parse.add_argument(
        "--numsThreshold",
        dest="nms_thre",
        help="Threshold for NMs",
        type=float,
        default=0.4
    )
    parse.add_argument(
        "--resolution",
        dest="res",
        help="Resolution to resize input images, lower to increase speed, higher to increase accuracy",
        default=416
    )

    return parse.parse_args()


parse = add_parse()
imgs = parse.imgs
outdet = parse.det
cfg = parse.cfg
weights = parse.weights
batch_size = int(parse.bs)
confidence = float(parse.conf)
nms_threshold = float(parse.nms_thre)
resolution = parse.res

CUDA = torch.cuda.is_available()
classes = load_names("./data/coco.names")
num_class = 80

# set up neural networks
print("Loading Network...")
model = DarkNet(cfg)
model.load_weights(weights)
print("Finish loading")

model.net_info["height"] = resolution
img_dim = int(resolution)
assert img_dim >= 32
assert img_dim % 32 == 0

if CUDA:
    model.cuda()

model.eval()

t_read_dir = time.time()
# get image list, 所有需要测试的图片集
try:
    img_list = [os.path.join(imgs, img) for img in os.listdir(imgs)]
except NotADirectoryError:
    img_list = [imgs]
except FileNotFoundError:
    print("Invalid image path, please check")
    exit()

# make output path
if not os.path.exists(outdet):
    os.makedirs(outdet)

t_load_img = time.time()
# load images as array
load_imgs = [cv2.imread(img) for img in img_list]

all_resize_img = list(map(prep_img4net, load_imgs, [
                      inp_size for x in range(len(load_imgs))]))

# list to store original image size
orig_img_dim = [(x.shape[1], x.shape[0]) for x in load_imgs]
# TODO: ??????
orig_img_dim = torch.FloatTensor(orig_img_dim).repeat(1, 2)

if CUDA:
    all_resize_img = all_resize_img.cuda()

batch_num = math.ceil(len(all_resize_img)/batch_size)

# get batch list
if batch_size != 1:
    img_batch = [torch.cat(all_resize_img[i*batch_size: (i+1)*batch_size])
                 for i in range(batch_num)]

write = 0
t_detection_start_time = time.time()

for i, batch in enumerate(img_batch):
    start = time.time()

    # utilize GPUs for computation
    if CUDA:
        batch = batch.cuda()

    prediction = model(batch, CUDA)

    prediction = write_results(
        prediction, confidence, num_class, nms_threshold)

    end = time.time()

    if isinstance(prediction, int):
        # 未检测出任何结果，输出为0
        for img_num, img in enumerate(img_list[i*batch_size:(i+1)*batch_size]):
            img_id = i*batch_size + img_num
            print("{} predicted {} seconds".format(
                img.split("/")[-1], (end-start)/batch_size))
            print("{0:20s} {1:s}".format("Object detected: ", ""))
            print("-"*50)

        continue

    prediction[:, 0] += i*batch_size

    if not write:
        output = prediction
        write = 1
    else:
        output = torch.cat(output, prediction)

    for img_num, img in enumerate(img_list[i*batch_size:(i+1)*batch_size]):
        img_id = i*batch_size + img_num
        obj = [classes[int(x[-2])] for x in output if x[0] == img_id]
        print(obj)
        print("{} predicted {} seconds".format(
            img.split("/")[-1], (end-start)/batch_size))
        print("{0:20s} {1:s}".format("Object detected: ", "".join(obj)))
        print("-"*50)

    if CUDA:
        # makes sure that CUDA kernel is synchronized with the CPU.
        # Otherwise, CUDA kernel returns the control to CPU as soon as
        # the GPU job is queued and well before the GPU job is completed
        torch.cuda.synchronize()

    try:
        output
    except NameError:
        print("No objects detected")
        exit()

orig_img_dim = torch.index_select(orig_img_dim, 0, output[:, 0].long())

# 获取原图与padded图片的形变系数，形变只发生在w,h中的一边
scale_fac = torch.min(img_dim/orig_img_dim, 1)[0].view(-1, 1)

# output中的坐标转化为padded图片中有图部分的相对坐标
output[:, [1, 3]] -= (img_dim - scale_fac*orig_img_dim[:, 0]).view(-1, 1)/2
output[:, [2, 4]] -= (img_dim - scale_fac*orig_img_dim[:, 1]).view(-1, 1)/2

# 将padded中的图片部分恢复到输入的原图尺寸
output[:, 1:5] /= scale_fac

for i in range(output.shape[0]):
    output[i, [1, 3]] = torch.clamp(output[i, [1, 3]], 0., orig_img_dim[i, 0])
    output[i, [2, 4]] = torch.clamp(output[i, [2, 4]], 0., orig_img_dim[i, 1])

t_class_load = time.time()
colors = pkl.load(open("pallete", "rb"))

t_draw = time.time()


def draw_box(x, result):
    c1 = tuple(x[1:3].int())
    c2 = tuple(x[3:5].int())
    img = result[int(x[0])]
    label = "{0}".format(classes[x[-2]])
    color = random.choice(colors)
    cv2.rectangle(img, c1, c2, color, thickness=1)

    t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
    c2 = (c1[0]+t_size[0]+3, c1[1]+t_size[1]+4)
    cv2.rectangle(img, c1, c2, color, -1)   # filled color

    cv2.putText(img, label, (c1[0], c2[1]),
                cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 1)

    return img


list(map(lambda x: (x, load_imgs), output))

det_img_names = pd.Series(img_list).apply(
    lambda x: "{}/det_{}".format(outdet, x.split("/")[-1]))

list(map(cv2.imwrite, det_img_names, load_imgs))

t_end = time.time()

print("SUMMURY")
print("-"*50)
print("{:25s} {}".format("Task", "Process time(in second)"))
print()
print("{:25s} {2.3f}".format("Reading address", t_load_img-t_read_dir))
print("{:25s} {2.3f}".format("Loading batch", t_detection_start_time-t_load_img))
print("{:25s} {2.3f}".format("Detection ("+str(len(img_list)+") images"),
                             t_class_load-t_detection_start_time))
print("{:25s} {2.3f}".format("Drawing box", t_end-t_draw))
print("{:25s} {2.3f}".format("Average time per image",
                             (t_end-t_load_img)/len(img_list)))
print("-"*50)
torch.cuda.empty_cache()
