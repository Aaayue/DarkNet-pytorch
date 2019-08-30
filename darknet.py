from __future__ import division

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import cv2

from util import *


class EmptyLayer(nn.Module):
    def __init__(self):
        super(EmptyLayer, self).__init__()


class DetectionLayer(nn.Module):
    def __init__(self, anchors):
        super(DetectionLayer, self).__init__()
        self.anchors = anchors


def get_input(imgfile):
    img = cv2.imread(imgfile)
    img = cv2.resize(img, (608, 608))
    img = img[:, :, :: -1].transpose((2, 0, 1))
    # cv2 read channel: BGR
    # BGR -> RBG | HWC -> CHW
    img = img[np.newaxis, :, :, :] / 225.
    # add new axis for batch
    img = torch.tensor(img, requires_grad=True)
    return img


def parse_cfg(cfgfile):
    """
    Takes a configuration file

    Returns a list of blocks. Each blocks describes a block in the neural
    network to be built. Block is represented as a dictionary in the list
    """
    txt = open(cfgfile, "r")
    lines = txt.read().split("\n")
    lines = [x for x in lines if len(x) != 0]
    lines = [x for x in lines if x[0] != "#"]
    lines = [x.lstrip().rstrip() for x in lines]

    block = {}
    blocks = []

    for line in lines:
        if line[0] == "[":
            if len(block) != 0:
                blocks.append(block)
                block = {}
            block["type"] = line[1:-1].rstrip()
        else:
            key, value = line.split("=")
            block[key.rstrip()] = value.lstrip()
    blocks.append(block)

    return blocks


def create_module(blocks):
    """
    Create modules from blocks list sequencially
    """
    net_info = blocks[0]
    module_list = nn.ModuleList()
    prev_filters = 3
    out_filters = []

    for index, x in enumerate(blocks[1:]):
        module = nn.Sequential()

        if x["type"] == "convolutional":
            try:
                batch_norm = int(x["batch_normalize"])
                bias = False
            except Exception:
                batch_norm = 0
                bias = True
            filters = int(x["filters"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            padding = int(x["pad"])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0
            activation = x["activation"]
            conv2d = nn.Conv2d(prev_filters, filters,
                               kernel_size, stride, pad, bias=bias)
            module.add_module("conv_{0}".format(index), conv2d)

            if batch_norm:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == "leaky":
                act = nn.LeakyReLU(0.1, inplace=True)
                module.add_module("leaky_{0}".format(index), act)

        elif x["type"] == "upsample":
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor=stride, mode="bilinear")
            module.add_module("upsample_{0}".format(index), upsample)

        elif x["type"] == "route":
            x["layers"] = x["layers"].split(",")
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if end > 0:
                end = end - index
            if start > 0:
                start = start - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)

            if end == 0:
                filters = out_filters[start + index]
            else:
                filters = out_filters[start + index] + out_filters[end + index]

        elif x["type"] == "shortcut":
            skip = EmptyLayer()
            module.add_module("shortcut_{0}".format(index), skip)

        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(i) for i in mask]

            anchors = x["anchors"].split(",  ")
            anchors = [a.split(",") for a in anchors]
            anchors = [[int(i) for i in j] for j in anchors]
            anchors = [anchors[i] for i in mask]

            yolo = DetectionLayer(anchors)
            module.add_module("yolo_{0}".format(index), yolo)

        prev_filters = filters
        out_filters.append(filters)
        module_list.append(module)

    return net_info, module_list


class DarkNet(nn.Module):
    """
    build darknet, manage the output and input of the network
    """

    def __init__(self, cfgfile):
        super(DarkNet, self).__init__()
        self.blocks = parse_cfg(cfgfile)
        self.net_info, self.module_list = create_module(self.blocks)

    def forward(self, x, CUDA):
        modules = self.blocks[1:]
        output = {}

        write = 0
        # Indicate whether we have encountered the first detection or not
        for i, module in enumerate(modules):
            if module["type"] == "convolutional" or module["type"] == "upsample":
                x = self.module_list[i](x)

            elif module["type"] == "route":
                # feature map output layer
                layers = [int(a) for a in module["layers"]]
                if layers[0] > 0:
                    layers[0] = layers[0] - i
                if len(layers) == 1:
                    x = output[i + layers[0]]
                else:
                    if layers[1] > 0:
                        layers[1] = layers[1] - i
                    x = torch.cat(
                        (output[i + layers[0]], output[i + layers[1]]), 1)
                    # input and output of a convolutional layer has
                    # the format `B x C x H x W`

            elif module["type"] == "shortcut":
                # output of residual net
                fromm = int(module["from"])
                x = output[i - 1] + output[i + fromm]

            elif module["type"] == "yolo":
                anchors = self.module_list[i][0].anchors
                inp_img = int(self.net_info["height"])
                num_classes = int(module["classes"])

                x = x.data
                x = predict_transform(x, inp_img, anchors, num_classes, CUDA)

                if not write:
                    detection = x
                    write = 1
                else:
                    detection = torch.cat((detection, x), 1)

            output[i] = x
        # output the concatenate of three different yolo output layers
        return detection

    def load_weights(self, weightfile):

        with open(weightfile, "rb") as fb:
            # the first 5 int32 values are constitute the header
            header = np.fromfile(fb, dtype=np.int32, count=5)
            self.header = torch.from_numpy(header)
            self.seen = self.header[3]
            weights = np.fromfile(fb, dtype=np.float32)

        prt = 0
        for i, block in enumerate(self.blocks[1:]):
            module = self.module_list[i]

            if block["type"] == "convolutional":
                conv = module[0]
                try:
                    batch_norm = int(block["batch_normalize"])
                except Exception:
                    batch_norm = 0

                if batch_norm:
                    bn = module[1]
                    num_bu_biases = bn.bias.numel()
                    # get the size of module.bias

                    bn_bias = torch.from_numpy(
                        weights[prt: prt + num_bu_biases]
                    ).view_as(bn.bias.data)
                    prt += num_bu_biases

                    bn_weight = torch.from_numpy(
                        weights[prt: prt + num_bu_biases]
                    ).view_as(bn.weight.data)
                    prt += num_bu_biases

                    bn_running_mean = torch.from_numpy(
                        weights[prt: prt + num_bu_biases]
                    ).view_as(bn.running_mean)
                    prt += num_bu_biases

                    bn_running_var = torch.from_numpy(
                        weights[prt: prt + num_bu_biases]
                    ).view_as(bn.running_var)
                    prt += num_bu_biases

                    bn.bias.data.copy_(bn_bias)
                    bn.weight.data.copy_(bn_weight)
                    bn.running_mean.copy_(bn_running_mean)
                    bn.running_var.copy_(bn_running_var)

                else:
                    num_conv_bias = conv.bias.numel()
                    conv_bias = torch.from_numpy(
                        weights[prt: prt + num_conv_bias]
                    ).view_as(conv.bias.data)
                    prt += num_conv_bias

                    conv.bias.data.copy_(conv_bias)

                num_conv_weight = conv.weight.numel()
                conv_weight = torch.from_numpy(
                    weights[prt: prt + num_conv_weight]
                ).view_as(conv.weight.data)
                prt += num_conv_weight

                conv.weight.data.copy_(conv_weight)


if __name__ == "__main__":
    import pprint
    cfg = "/home/zy/Work/darknet-pyTorch/yolov3.cfg"
    img = "/home/zy/Work/darknet-pyTorch/dog-cycle-car.png"
    wgt = "/home/zy/Work/darknet-pyTorch/yolov3.weights"
    blocks = parse_cfg(cfg)
    net, module = create_module(blocks)
    save_json(net, "net-info.json")
    save_json(module, "module-list.json")
    pprint.pprint(blocks)
    pprint.pprint(net)
    pprint.pprint(module)
    net = DarkNet(cfg)
    img = get_input(img).float()
    res = net.forward(img, False)
    pprint.pprint(res.size())
    net.load_weights(wgt)
    input("stop here!")
