#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSORT 目标追踪脚本 - CPU 版本
使用 YOLOv3 检测器 + DeepSORT 追踪器
"""

import os
import cv2
import numpy as np
import tensorflow as tf
from deep_sort_realtime.deepsort_tracker import DeepSort

# 配置路径
YOLO_WEIGHTS = "/app/deep_sort_model/yolov3.weights"
YOLO_CFG = "/app/deep_sort_yolov3/model_data/yolov3.cfg"
DEEPSORT_MODEL = "/app/deep_sort_model/mars-small128.pb"
MOT17_DIR = "/mnt/MOT17"

def load_yolo_detector():
    """加载 YOLOv3 检测器"""
    print("加载 YOLOv3 检测器...")
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    # 使用 CPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("YOLOv3 加载完成")
    return net

def get_yolo_outputs(image, net):
    """使用 YOLOv3 检测目标"""
    height, width = image.shape[:2]

    # 创建 blob
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)

    # 前向传播
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    # 处理输出
    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]

            if confidence > 0.5:  # 置信度阈值
                # 检测框坐标
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))

    # 非极大值抑制
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    detections = []
    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            # 格式：[left, top, width, height, confidence]
            detections.append(([x, y, w, h], confidences[i], 'person'))

    return detections

def run_tracking(video_path, output_path=None):
    """运行 DeepSORT 追踪"""
    print(f"处理视频：{video_path}")

    # 加载检测器
    net = load_yolo_detector()

    # 初始化 DeepSORT 追踪器
    tracker = DeepSort(
        max_age=30,
        n_init=3,
        nms_max_overlap=1.0,
        max_cosine_distance=0.2,
        nn_budget=100,
        embedder="tfmobilenet",  # 使用轻量级模型
        half=False,
        bgr=True,
        embedder_gpu=False,
        embedder_wsize=DEEPSORT_MODEL,
        polygon=None,
        override_track_class=None
    )

    # 打开视频
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频：{video_path}")
        return

    # 视频属性
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"视频：{frame_width}x{frame_height}, {fps}fps, {total_frames}帧")

    # 视频写入器
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_count += 1

        # 每 5 帧检测一次（加速处理）
        if frame_count % 5 == 0:
            detections = get_yolo_outputs(frame, net)

            # 更新追踪器
            tracks = tracker.update_tracks(detections, frame=frame)
        else:
            tracks = tracker.update_tracks([], frame=frame)

        # 绘制追踪结果
        for track in tracks:
            if not track.is_confirmed():
                continue

            track_id = track.track_id
            ltrb = track.to_ltrb()  # left, top, right, bottom

            x1, y1, x2, y2 = map(int, ltrb)

            # 绘制边界框
            color = (0, 255, 0)  # 绿色
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # 绘制 ID
            label = f"ID: {track_id}"
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        # 显示帧信息
        info = f"Frame: {frame_count}/{total_frames}, Tracks: {len([t for t in tracks if t.is_confirmed()])}"
        cv2.putText(frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 写入输出
        if output_path:
            out.write(frame)

        # 显示进度
        if frame_count % 100 == 0:
            progress = 100 * frame_count / total_frames
            print(f"进度：{progress:.1f}% ({frame_count}/{total_frames})")

    cap.release()
    if output_path:
        out.release()

    print(f"完成！输出：{output_path}")

def test_simple_tracking():
    """简单测试 - 使用 MOT17 数据"""
    # 检查 MOT17 数据
    test_seq = os.path.join(MOT17_DIR, "MOT17-02", "det", "det.txt")
    if os.path.exists(test_seq):
        print(f"找到 MOT17 数据：{test_seq}")
        # 读取检测数据
        detections = np.loadtxt(test_seq, delimiter=',')
        print(f"检测数据形状：{detections.shape}")

    # 查找视频文件
    video_candidates = []
    for root, dirs, files in os.walk(MOT17_DIR):
        for f in files:
            if f.endswith(('.mp4', '.avi', '.mkv')):
                video_candidates.append(os.path.join(root, f))

    if video_candidates:
        print(f"找到视频文件：{video_candidates[0]}")
        output_video = "/app/output_tracking.mp4"
        run_tracking(video_candidates[0], output_video)
    else:
        print("未找到视频文件，使用摄像头测试...")
        # 可以使用摄像头测试
        # run_tracking(0, None)
        pass

if __name__ == "__main__":
    test_simple_tracking()
