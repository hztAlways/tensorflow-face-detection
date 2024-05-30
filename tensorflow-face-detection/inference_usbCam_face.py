#!/usr/bin/python
# -*- coding: utf-8 -*-
# pylint: disable=C0103
# pylint: disable=E1101

import sys
import time
import numpy as np
import tensorflow as tf
import cv2

from utils import label_map_util
from utils import visualization_utils_color as vis_util

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_CKPT = './model/frozen_inference_graph_face.pb'
# 冻结检测图的路径，这是用于目标检测的实际模型

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = './protos/face_label_map.pbtxt'
# 用于为每个框添加正确标签的字符串列表的路径

NUM_CLASSES = 2
# 类别数量

# 加载标签映射
label_map = label_map_util.load_labelmap(PATH_TO_LABELS)
# 转换标签映射为类别列表
categories = label_map_util.convert_label_map_to_categories(label_map, max_num_classes=NUM_CLASSES,
                                                            use_display_name=True)
# 创建类别索引字典
category_index = label_map_util.create_category_index(categories)


# 定义 TensorflowFaceDetector 类
class TensoflowFaceDector(object):
    def __init__(self, PATH_TO_CKPT):
        """Tensorflow detector
        """
        # 初始化检测图
        self.detection_graph = tf.Graph()
        with self.detection_graph.as_default():
            # 创建一个 GraphDef 对象
            od_graph_def = tf.compat.v1.GraphDef()
            # 读取冻结检测图
            with tf.io.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
                serialized_graph = fid.read()
                # 将图定义从字符串解析到 od_graph_def 中
                od_graph_def.ParseFromString(serialized_graph)
                # 导入图定义
                tf.import_graph_def(od_graph_def, name='')

        with self.detection_graph.as_default():
            # 配置 GPU 选项
            config = tf.compat.v1.ConfigProto()
            config.gpu_options.allow_growth = True
            # 创建会话
            self.sess = tf.compat.v1.Session(graph=self.detection_graph, config=config)
            # 初始化窗口设置标志
            self.windowNotSet = True

    # 定义 run 方法，用于运行目标检测
    def run(self, image):
        """image: bgr image
        return (boxes, scores, classes, num_detections)
        """
        # 将 BGR 图像转换为 RGB
        image_np = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # 扩展维度以匹配模型输入要求 [1, None, None, 3]
        image_np_expanded = np.expand_dims(image_np, axis=0)
        # 获取输入张量
        image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
        # 获取输出张量
        boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
        scores = self.detection_graph.get_tensor_by_name('detection_scores:0')
        classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
        num_detections = self.detection_graph.get_tensor_by_name('num_detections:0')
        # 进行检测
        start_time = time.time()
        (boxes, scores, classes, num_detections) = self.sess.run(
            [boxes, scores, classes, num_detections],
            feed_dict={image_tensor: image_np_expanded})
        elapsed_time = time.time() - start_time
        print('inference time cost: {}'.format(elapsed_time))
        # 返回检测结果
        return (boxes, scores, classes, num_detections)


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("usage:%s (cameraID | filename) Detect faces\
 in the video example:%s 0" % (sys.argv[0], sys.argv[0]))
        exit(1)

    try:
        # 尝试将输入参数转换为摄像头 ID
        camID = int(sys.argv[1])
    except:
        # 如果转换失败，将输入参数视为文件名
        camID = sys.argv[1]

    # 创建 TensorflowFaceDector 对象
    tDetector = TensoflowFaceDector(PATH_TO_CKPT)

    # 打开视频捕获设备（摄像头或视频文件）
    cap = cv2.VideoCapture(camID)
    windowNotSet = True
    while True:
        # 读取视频帧
        ret, image = cap.read()
        if ret == 0:
            break

        [h, w] = image.shape[:2]
        print(h, w)
        # 镜像翻转图像
        image = cv2.flip(image, 1)

        # 运行检测器
        (boxes, scores, classes, num_detections) = tDetector.run(image)

        # 在图像上绘制检测结果
        vis_util.visualize_boxes_and_labels_on_image_array(
            image,
            np.squeeze(boxes),
            np.squeeze(classes).astype(np.int32),
            np.squeeze(scores),
            category_index,
            use_normalized_coordinates=True,
            line_thickness=4)

        if windowNotSet is True:
            # 设置窗口属性
            cv2.namedWindow("tensorflow based (%d, %d)" % (w, h), cv2.WINDOW_NORMAL)
            windowNotSet = False

        # 显示图像
        cv2.imshow("tensorflow based (%d, %d)" % (w, h), image)
        k = cv2.waitKey(1) & 0xff
        if k == ord('q') or k == 27:
            break

    cap.release()
    # 释放视频捕获设备
