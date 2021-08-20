# coding: utf-8
from __future__ import division, print_function
import tensorflow as tf
import numpy as np
import argparse
import cv2
from utils.misc_utils import parse_anchors, read_class_names
from utils.nms_utils import gpu_nms
from utils.plot_utils import get_color_table, plot_one_box
from utils.data_aug import letterbox_resize
from model import yolov3
import time
from absl import app, flags, logging
from flask import Flask, request, jsonify, make_response,json
from flask_cors import CORS
from io import BytesIO
from PIL import Image
import io
app = Flask(__name__)
CORS(app)

def parse_args():
    global color_table
    parser = argparse.ArgumentParser(description="YOLO-V3 test single image test procedure.")
    #parser.add_argument("input_image", type=str,
                        #help="The path of the input image.")
    parser.add_argument("--anchor_path", type=str, default="./data/my_data/yolo_anchors.txt",
                        help="The path of the anchor txt file.")
    parser.add_argument("--new_size", nargs='*', type=int, default=[416, 416],
                        help="Resize the input image with `new_size`, size format: [width, height]")
    parser.add_argument("--letterbox_resize", type=lambda x: (str(x).lower() == 'true'), default=True,
                        help="Whether to use the letterbox resize.")
    parser.add_argument("--class_name_path", type=str, default="./data/my_data/Screw_data.names",
                        help="The path of the class names.")
    parser.add_argument("--restore_path", type=str, default="./data/darknet_weights/yolov3-Screwdata.ckpt",
                        help="The path of the weights to restore.")
    args = parser.parse_args()
    args.anchors = parse_anchors(args.anchor_path)
    args.classes = read_class_names(args.class_name_path)
    args.num_class = len(args.classes)
    color_table = get_color_table(args.num_class)
    return args




@app.route('/predict', methods=['POST'])
def predict():
    if request.method =='POST':
        image = request.files['image'].read()
    img_ori = Image.open(io.BytesIO(image))
    img_ori = np.asarray(img_ori)
    args = parse_args()
    if args.letterbox_resize:
        img, resize_ratio, dw, dh = letterbox_resize(img_ori, args.new_size[0], args.new_size[1])
    else:
        height_ori, width_ori = img_ori.shape[:2]
        img = cv2.resize(img_ori, tuple(args.new_size))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.asarray(img, np.float32)
    img = img[np.newaxis, :] / 255.

    with tf.Session() as sess:
        input_data = tf.placeholder(tf.float32, [1, args.new_size[1], args.new_size[0], 3])
        yolo_model = yolov3(args.num_class, args.anchors)
        with tf.variable_scope('yolov3'):
            pred_feature_maps = yolo_model.forward(input_data, False)
        pred_boxes, pred_confs, pred_probs = yolo_model.predict(pred_feature_maps)

        pred_scores = pred_confs * pred_probs

        boxes, scores, labels = gpu_nms(pred_boxes, pred_scores, args.num_class, max_boxes=200, score_thresh=0.25, nms_thresh=0.6)

        saver = tf.train.Saver()
        saver.restore(sess, args.restore_path)

        t1 = time.time()
        boxes_, scores_, labels_ = sess.run([boxes, scores, labels], feed_dict={input_data: img})
        t2 = time.time()
        print('time: {}'.format(t2 - t1))

        # rescale the coordinates to the original image
        if args.letterbox_resize:
            boxes_[:, [0, 2]] = (boxes_[:, [0, 2]] - dw) / resize_ratio
            boxes_[:, [1, 3]] = (boxes_[:, [1, 3]] - dh) / resize_ratio

        else:
            boxes_[:, [0, 2]] *= (width_ori/float(args.new_size[0]))
            boxes_[:, [1, 3]] *= (height_ori/float(args.new_size[1]))


        boxes_ = ' '.join([str(elem) for elem in boxes_])
        scores_ = ' '.join([str(elem) for elem in scores_])
        labels_ = ' '.join([str(elem) for elem in labels_])
        x2 = boxes_[1:4]
        x1 = boxes_[11:14]
        y2 = boxes_[21:24]
        y1 = boxes_[31:34]

        
        classification={
            "class name": args.classes
        }

        response = {
            "label": labels_,
            "probability": scores_,
            'x_max': x2,
            'x_min': x1,
            'y_max': y2,
            'y_min': y1,
            }
            
    return jsonify( classification, {"predictions": response})

if __name__ == '__main__':
    app.run('localhost', port=5555, debug=True)
