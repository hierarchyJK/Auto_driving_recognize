# -*-coding:utf-8 -*-
"""
@project:untitled3
@author:Kun_J
@file:.py
@ide:untitled3
@time:2019-01-22 18:12:32
@month:一月
"""
import argparse
import os
import matplotlib.pyplot as plt
import scipy.io
import scipy.misc
import numpy as np
import pandas as pd
import PIL
import tensorflow as tf
from keras import backend as K
from keras.layers import Input,Lambda,Conv2D
from keras.models import load_model,Model
from yolo_utils import read_classes, read_anchors, generate_colors, preprocess_image, draw_boxes, scale_boxes
from keras_yolo import yolo_head, yolo_boxes_to_corners, preprocess_true_boxes, yolo_loss, yolo_body


def yolo_filter_boxes(box_confidence, boxes, box_class_probs, thresthod = .6):
    """
    Filters YOLO boxes by thresholding on object and class confidence.
    :param box_confidence: --tensor of shape (19, 19, 5,1)
    :param boxes: -- tensor of shape (19, 19, 5, 4) (后面用到的是边角corner coordinate)
    :param box_class_probs: -- tensor of shape (19, 19, 5, 80)
    :param thresthod: -- real value, if [highest class probability score < threshold],then get rid of the corresponding box]
    :return:
     scores -- tensor of shape(None, ),containing the class probability score for selected boxes
     boxes -- tensor of shape(None, 4),containing(b_x, b_y, b_h, b_w) coordinates of selected boxes
     classes -- tensor of shape(None, ),containing the index of the class detected by the selected boxes
    """
    ## First step：计算锚框的得分
    box_scores = box_confidence * box_class_probs
    ## Second step：找到最大值的锚框索引以及对应的最大值的锚框
    box_classes = K.argmax(box_scores,axis=-1)
    box_class_scores = K.max(box_scores,axis=-1)
    ## Third step：根据阈值创建掩码
    filtering_mask = (box_class_scores>=thresthod)
    ## 对scores， boxes 以及classes使用掩码
    scores = tf.boolean_mask(box_class_scores,filtering_mask)
    boxes = tf.boolean_mask(boxes, filtering_mask)
    classes = tf.boolean_mask(box_classes, filtering_mask)

    return scores, boxes, classes
# def yolo_filter_boxes_test():
#     with tf.Session() as test_a:
#         box_confidence = tf.random_normal([19, 19, 5, 1], mean=1, stddev=4, seed=1)
#         boxes = tf.random_normal([19, 19, 5, 4], mean=1, stddev=4, seed=1)
#         box_class_probs = tf.random_normal([19, 19, 5, 80], mean=1, stddev=4, seed=1)
#         scores, boxes, classes = yolo_filter_boxes(box_confidence, boxes, box_class_probs, thresthod=0.5)
#         print("scores[2] = " + str(scores[2].eval()))
#         print("boxes[2] = " + str(boxes[2].eval()))
#         print("classes[2] = " + str(classes[2].eval()))
#         print("scores.shape = " + str(scores.shape))
#         print("boxes.shape = " + str(boxes.shape))
#         print("classes.shape = " + str(classes.shape))
#         test_a.close()
##yolo_filter_boxes_test()

def iou(box1, box2):
    """
    实现两个锚框的交并比的计算
    :param box1: 第一个锚框，shape(x1,y1,x2,y2)
    :param box2: 第二个锚框，shape(x1,y1,x2,y2)
    :return:
    iou:实数，交并比
    """
    # 计算相交的区域的面积
    xi1 = np.maximum(box1[0], box2[0])
    yi1 = np.maximum(box1[1], box2[1])
    xi2 = np.minimum(box1[2], box2[2])
    yi2 = np.minimum(box1[3], box2[3])
    inter_area = (xi1 - xi2) * (yi1 - yi2)

    # 计算并集 Union(A,B) = A + B - Inter(A, B)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union_area = box1_area + box2_area - inter_area

    # 计算交并比
    iou = inter_area / union_area

    return iou
def iou_test():
    box1 = (2,1,4,3)
    box2 = (1,2,3,4)
    print("iou = " + str(iou(box1, box2)))
##iou_test()

"""实现非最大值抑制函数：
1：选择分值高度额锚框
2：计算与其他框的重叠部分，并删除与该锚框交叠较大的网格
3：返回第一步，直到不再有比当前选中的框得分更低的框
Note：这将删除与选定框有较大重叠的其他所有锚框，只有得分最高的锚框仍然存在"""

def yolo_non_max_suppression(scores, boxes, classes, max_boxes = 10, iou_threshold = 0.5):
    """
    Applies Non-max suppression (NMS) to set of boxes
    Implement yolo_non_max_suppression using Tensorflow
    :param scores: tensor类型，(None, ),yolo_filter_boxes()的输出
    :param boxes: tensor类型，(None,4),yolo_filter_boxes()的输出
    :param classes: tensor类型，(None, ),yolo_filter_boxes()的输出
    :param max_boxes: Integer,预测锚框数量的最大值
    :param iou_threshold: real value，交并比阈值
    :return:
    scores: tensor,( ,None),predicted score for each box
    boxes: tensor,(4,None),predicted box coordinates
    classes: tensor,( ,None),predicted class for each box
    Note:The "None" dimension of the output tensors has obviously to be less than max_boxes. Note also that this
    function will transpose the shapes of scores, boxes, classes. This is made for convenience.
    """
    # 用于te.image.non_max_suppression()
    max_boxes_tensor = K.variable(max_boxes, dtype="int32")
    # 初始化变量max_boxes_tensor
    K.get_session().run(tf.variables_initializer([max_boxes_tensor]))
    # 使用tf.image.non_max_suppression()来获取我们保留框对应的索引列表
    nms_indices = tf.image.non_max_suppression(boxes, scores, max_boxes,iou_threshold)

    # 使用K.gather()来选择保留的锚框
    scores = K.gather(scores, nms_indices)
    boxes = K.gather(boxes, nms_indices)
    classes = K.gather(classes, nms_indices)

    return scores, boxes, classes
def yolo_non_max_suppression_test():
    with tf.Session() as test_b:
        scores = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
        boxes = tf.random_normal([54,4],mean=1, stddev=4, seed=1)
        classes = tf.random_normal([54, ], mean=1, stddev=4, seed=1)
        scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes)
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.eval().shape))
        print("boxes.shape = " + str(boxes.eval().shape))
        print("classes.shape = " + str(classes.eval().shape))
#yolo_non_max_suppression_test()
def yolo_eval(yolo_outputs, image_shape=(720.,1280.), max_boxes=10, score_threshold=0.6, iou_threshold=0.5):
    """
    将YOLO编码的输出（很多框）转换为预测框以及他们的分数、框坐标和类
    :param yolo_outputs: 编码模型的输出（对于维度为608*608*3的图片），包含4个tensor类型的变量：
                          box_confidence:tensor类型，shape of (None,19,19,5,1)
                          box_xy:tensor类型，shape of (None,19,19,5,2)
                          box_wh:tensor类型，shape of (None,19,19,5,2)
                          box_class_probs:tensor类型， shape of (None,19,19,5,80)
    :param image_shape:tensor类型，shape of (2, )，包含了输入的图像的维度，这里是(608, 608)
    :param max_boxes:integer,预测的锚框数量的最大值
    :param score_threshold:real value，可能的阈值
    :param iou_threshold:real value,交并比阈值
    :return:
            scores:tensor类型，shape of (None, ),每个锚框的预测的可能值
            boxes:tensor类型，shape of (None,4),预测锚框的坐标
            classes:tensor类型，shape of (None, ),每个锚框的预测的分类
    """
    # 获取YOLO模型的输出
    box_confidence, box_xy, box_wh, box_class_probs = yolo_outputs

    # 中心点转换为边角
    boxes = yolo_boxes_to_corners(box_xy, box_wh)

    # score过滤，第一个过滤器
    scores, boxes, classes = yolo_filter_boxes(box_confidence,boxes,box_class_probs, score_threshold)

    # 缩放锚框，以适应原始图像
    boxes = scale_boxes(boxes, image_shape)

    # 使用非最大值抑制，第二个过滤器
    scores, boxes, classes = yolo_non_max_suppression(scores, boxes, classes, max_boxes, iou_threshold)
    return scores, boxes, classes
def yolo_eval_test():
    with tf.Session() as sess:
        yolo_outputs = (tf.random_normal([19,19,5,1],mean=1,stddev=4,seed=1),
                        tf.random_normal([19,19,5,2],mean=1,stddev=4,seed=1),
                        tf.random_normal([19,19,5,2],mean=1,stddev=4,seed=1),
                        tf.random_normal([19,19,5,80],mean=1,stddev=4,seed=1))
        scores, boxes, classes = yolo_eval(yolo_outputs)
        print("scores[2] = " + str(scores[2].eval()))
        print("boxes[2] = " + str(boxes[2].eval()))
        print("classes[2] = " + str(classes[2].eval()))
        print("scores.shape = " + str(scores.eval().shape))
        print("boxes.shape = " + str(boxes.eval().shape))
        print("classes.shape = " + str(classes.eval().shape))
#yolo_eval_test()
"""
对YOLO的总结：
1、输入图像为(608, 608)
2、输入的图像先要经过一个CNN模型，返回一个(19, 19, 5, 85)的输出
3、再对最后的两维降维，输出变成(19, 19, 5, 425):
    ·每个19*19的单元格拥有425个数字
    ·425=5*85，即每个单元格拥有5个锚框，每个锚框由5个基本信息+80个分类预测构成
    ·85=5+80，其中5个基本信息是(Pc,Px,Py,Ph,Pw)，剩下的80个就是80个分类预测
4、然后我们会根据一下规则选择锚框：
    ·预测分数阈值：丢弃分数低于阈值的分类的锚框
    ·非最大值抑制：计算交并比，并避免选择重叠的框
5、最后给出YOLO的输出
"""
sess = K.get_session()
class_names = read_classes('F:\\吴恩达DL作业\课后作业\\代码作业\\第四课第三周编程作业\\Car detection for Autonomous Driving\\model_data\\coco_classes.txt')
anchors = read_anchors('F:\\吴恩达DL作业\课后作业\\代码作业\\第四课第三周编程作业\\Car detection for Autonomous Driving\\model_data\\yolo_anchors.txt')
image_shape = (720., 1280.)
yolo_model = load_model('F:\\吴恩达DL作业\课后作业\\代码作业\\第四课第三周编程作业\\Car detection for Autonomous Driving\\model_data\\yolo.h5')
yolo_model.summary()

yolo_outputs = yolo_head(yolo_model.output, anchors, len(class_names))
scores, boxes, classes = yolo_eval(yolo_outputs, image_shape)

def predict(sess, image_file, is_show_info=True, is_plot=True):
    """
    运行存储在sess的计算图以预测image_file的边界框，打印出预测图与信息
    :param sess: 包含了YOLO计算图的TensorFlow/keras的会话
    :param imagefile: 存储images文件下的图片名称
    :param is_show_info:
    :param is_plot:
    :return:
            out_scores:tensor, (None, ),锚框的预测的可能值
            out_boxes:tensor, (None,4),包含了锚框位置信息
            out_classes:tensor, (None, ),锚框的预测的分类索引
    """
    image, image_data = preprocess_image(image_file, model_image_size =(608, 608))###预处理图像
    out_scores, out_boxes, out_classes = sess.run([scores,boxes,classes],feed_dict={yolo_model.input:image_data, K.learning_phase():0})
    if is_show_info:
        print("在" + str(image_file)+"中找到"+str(len(out_boxes))+"个锚框。")
    colors = generate_colors(class_names)
    draw_boxes(image, out_scores, out_boxes, out_classes, class_names, colors)
    image.save(os.path.join('C:\\Users\\korey\\Desktop\\car',image_file), quality=90)
    if is_plot:
        out_image = plt.imread(os.path.join('C:\\Users\\korey\\Desktop\\car',image_file))
        plt.imshow(out_image)
        plt.show()
    return out_scores, out_boxes, out_classes

#out_scores, out_boxes, out_classes = predict(sess,'test.jpg')
# image_test = plt.imread('test.jpg')
# plt.imshow(image_test)
# plt.show()
rootdir = 'F:\\吴恩达DL作业\\课后作业\\代码作业\\第四课第三周编程作业\\Car detection for Autonomous Driving\\images'
for parent,dirnames,filenames in os.walk(rootdir):#1.父目录 2.所有文件夹名字（不含路径） 3.所有文件名字
    for filename in filenames:
        print('当前图片：'+str( os.path.join(parent, filename)))
        out_scores, out_boxes, out_classes = predict(sess, os.path.join(parent, filename))
