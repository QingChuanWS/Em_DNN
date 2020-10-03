# -*- coding:utf-8 -*-
""" python demo usage about MNN API """
from __future__ import print_function

import argparse
import time

import cv2
import MNN
import numpy as np
from PIL import Image

from utils.anchor_decode import decode_bbox
from utils.anchor_generator import generate_anchors
from utils.nms import single_class_non_max_suppression

model_file = '/home/hyliu/MNN_Mask/facemask.mnn'

# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

def inference(image,
              conf_thresh=0.8,
              iou_thresh=0.2,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size. 
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    output_info = []
    height, width, _ = image.shape
    interpreter = MNN.Interpreter(model_file)
    session = interpreter.createSession()
    input_tensor = interpreter.getSessionInput(session)
    # image = np.copy(image)
    image_f32 = np.float32(image)
    image_f32 = cv2.resize(image_f32, target_shape)
    image_f32 = image_f32.astype(float)
    image_f32 = image_f32 * (1/255.0, 1/255.0, 1/255.0)
    #preprocess it
    image_np = np.float32(image_f32)
    image_np = image_np.transpose((2, 0, 1))

    tmp_input = MNN.Tensor((1, 3, 260, 260), MNN.Halide_Type_Float,\
                    image_np, MNN.Tensor_DimensionType_Caffe)
    #construct tensor from np.ndarray
    input_tensor.copyFrom(tmp_input)
    interpreter.runSession(session)
    y_bboxes_output = interpreter.getSessionOutput(session,'loc_branch_concat').getData()
    y_cls_output = interpreter.getSessionOutput(session,'cls_branch_concat').getData()
    
    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)
    
    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                    bbox_max_scores,
                                                    conf_thresh=conf_thresh,
                                                    iou_thresh=iou_thresh,
                                                    )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0 , 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin-2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Face Mask Detection")
    parser.add_argument('--img-path', type=str, help='path to your image.')
    # parser.add_argument('--hdf5', type=str, help='keras hdf5 file')
    args = parser.parse_args()
    imgPath = args.img_path
    img = cv2.imread(imgPath)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    inference(img, show_result=True, target_shape=(260, 260))
