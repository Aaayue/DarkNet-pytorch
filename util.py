from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import variable
import numpy as np
import cv2
import json


def predict_transform(prediction, inp_img, anchors, num_classes, CUDA=True):
    batch_size = prediction.size()[0]
    stride = inp_img // prediction.size()[2]
    grid_size = inp_img // stride
    bbox_attrs = 5 + num_classes
    num_anchors = len(anchors)

    prediction = prediction.view(
        batch_size, bbox_attrs*num_anchors, grid_size*grid_size)
    prediction = prediction.transpose(1, 2).contiguous()
    prediction = prediction.view(
        batch_size, grid_size*grid_size*num_anchors, bbox_attrs)

    anchors = [(a[0]/grid_size, a[1]/grid_size) for a in anchors]
    # The dimensions of the anchors are in accordance to the height
    # and width attributes of the net block.

    for i in (0, 1, 4):
        # Sigmoid for (x, y) and objectness score
        prediction[:, :, i] = torch.sigmoid(prediction[:, :, i])

    grid = np.arange(grid_size)
    x_offset, y_offset = np.meshgrid(grid, grid)
    x_offset = torch.FloatTensor(x_offset).view(-1, 1)
    y_offset = torch.FloatTensor(y_offset).view(-1, 1)

    if CUDA:
        x_offset = x_offset.cuda()
        y_offset = y_offset.cuda()

    x_y_offset = torch.cat((x_offset, y_offset), 1)
    x_y_offset = x_y_offset.repeat(1, num_anchors).view(-1, 2).unsqueeze(0)
    # x_y_offset.size(): (169, 2) -> (1, 507, 2)

    prediction[:, :, :2] += x_y_offset
    # add co-ordinate to the (x, y) of bbox

    anchors = torch.FloatTensor(anchors)
    if CUDA:
        anchors = anchors.cuda()

    anchors = anchors.repeat(grid_size * grid_size, 1).unsqueeze(0)
    prediction[:, :, 2:4] = torch.exp(prediction[:, :, 2:4]) * anchors

    prediction[:, :, 5:5 +
               num_classes] = torch.sigmoid(prediction[:, :, 5:5+num_classes])

    prediction[:, :, :4] *= stride
    # prediction.size(): batch_size x 13*13*3 x 85

    return prediction


def write_results(prediction, confidence, class_nums, nms_threshold=0.4):
    # filter bbox whose objectness score lower than confidence
    obj_mask = (prediction[:, :, 4] > confidence).float().unsqueeze(2)
    prediction = prediction * obj_mask

    # transform center coordinates to corner coordinates
    corner_box = prediction.new(prediction.shape)
    corner_box[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 3]/2
    corner_box[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 2]/2
    corner_box[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 3]/2
    corner_box[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 2]/2
    prediction[:, :, :4] = corner_box[:, :, :4]

    batch_size = prediction.shape[0]
    write = False   # ????

    for i in range(batch_size):
        # keep the right class id and score, [1, 22743, 85] -> [1, 22743, 7]
        # (Xmin, Ymin, Xmax, Ymax, class_id, class_score)
        img_pred = prediction[i]

        max_class_score, max_class = torch.max(img_pred[:, 5:], 0)
        max_class = max_class.float().unsqueeze(1)
        max_class_score = max_class_score.float().unsqueeze(1)
        img_pred = torch.cat((img_pred[:, :5], max_class, max_class_score), 1)

        nonzero_id = torch.nonzero(img_pred[:, 4])

        try:
            img_pred_ = img_pred[nonzero_id.squeeze(), :].view(-1, 7)
        except Exception:
            continue

        if img_pred_.shape[0] == 0:
            continue

        # get various classes detected in the image
        # TODO different
        img_classes = torch.unique(img_pred_[:, -2])

        for clss in img_classes:
            cls_mask = img_pred_ * \
                (img_pred_[:, -2] == clss).float().unsqueeze(1)
            cls_mask_idx = torch.nonzero(cls_mask[:, -1]).squeeze()
            cls_mask = cls_mask[cls_mask_idx].view(-1, 7)

            # sort objectness-score in this class
            _, obj_sort_idx = torch.sort(cls_mask[:, 4], desending=True)
            cls_sort_pred = cls_mask[obj_sort_idx]
            idx = cls_sort_pred.size(0)

            for i in range(idx):
                # iterate bboxs to get final bbox for each class

                try:
                    ious = bbox_iou(cls_sort_pred[i], cls_sort_pred[i+1:])
                except Exception as e:
                    print("IOU ERROR! {}".format(e))
                    break

                iou_mask = (ious < nms_threshold).float().unsqueeze(1)
                cls_sort_pred[i+1:] *= iou_mask

                non_zero_id = torch.nonzero(cls_sort_pred[:, 4]).squeeze()
                cls_sort_pred = cls_sort_pred[non_zero_id].view(-1, 7)

            # add img id in batch to the nms result
            img_id = cls_sort_pred.new(cls_sort_pred.size(0), 1).fill_(i)
            if not write:
                output = torch.cat((img_id, cls_sort_pred), 1)
                write = True
            else:
                out = torch.cat((img_id, cls_sort_pred), 1)
                output = torch.cat((output, out), 0)

            if write:
                # the output of write_results is (D, 8)
                # D is num of "true" detection bbox for every class in this batch;
                # 8 is [img_id in batch, coordinates*4, objextness_score, class_index, max_class_score]
                return output
            else:
                return 0


def bbox_iou(box1, box2):
    """
    calculate ious between boxes
    """

    b1_xmin, b1_ymin, b1_xmax, b1_ymax = box1[:, :3]
    b2_xmin, b2_ymin, b2_xmax, b2_ymax = box2[:, :3]

    inter_xmin = torch.max(b1_xmin, b2_xmin)
    inter_ymin = torch.max(b1_ymin, b2_ymin)
    inter_xmax = torch.min(b1_xmax, b2_xmax)
    inter_ymax = torch.min(b1_ymax, b2_ymax)

    inter_area = torch.clamp(inter_ymax - inter_ymin, min=0) * \
        torch.clamp(inter_xmax - inter_xmin, min=0)

    box1_area = (b1_xmax - b1_xmin) * (b1_ymax - b1_ymin)
    box2_area = (b2_xmax - b2_xmin) * (b2_ymax - b2_ymin)

    iou = inter_area / (box1_area + box2_area - inter_area)

    return iou


def load_names(file):
    with open(file, "r") as fp:
        names = fp.read().split("\n")[:-1]

    return names


def save_json(data, file):
    with open(file) as f:
        json.dump(data, f)

    return file


def letterbox_img(img, inp_size):
    """
    resize and padding image, keep the aspect ratio consistent
    """
    img_h, img_w = img.shape
    h, w = inp_size
    # resize and keep aspect ratio
    # w,h 有一边会保留原始尺寸
    new_h = int(img_h * min(h/img_h, w/img_w))
    new_w = int(img_w * min(h/img_h, w/img_w))

    new_img = cv2.resize(img, (new_h, new_w), interpolation=cv2.INTER_CUBIC)

    out_img = np.full((h, w, 3), 128)

    out_img[(h-new_h)/2: (h-new_h)/2+new_h, (w-new_w) /
            2: (w-new_w)/2+new_w, :] = new_img

    return out_img


def prep_img4net(img, inp_size):
    img = letterbox_img(img, inp_size)
    # cv2 get BGR, convert to RGB
    # transpose (height, width, channel) to (channel, height, width)
    img = img[:, :, ::-1].transpose((2, 0, 1)).copy()
    input_img = torch.from_numpy(img).float().div(255.).unsqueeze(0)

    return input_img
