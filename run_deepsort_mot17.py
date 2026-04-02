#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSORT 简单目标追踪 - MOT17 图像序列版本
CPU 可运行
"""

import os
import cv2
import numpy as np

# 配置路径
YOLO_WEIGHTS = "/app/deep_sort_model/yolov3.weights"
YOLO_CFG = "/app/deep_sort_yolov3/yolov3.cfg"
MOT17_SEQ = "/mnt/MOT17/train/MOT17-02-FRCNN/img1"
OUTPUT_DIR = "/app/output"

def load_yolo_detector():
    """加载 YOLOv3 检测器"""
    print("加载 YOLOv3 检测器...")
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("YOLOv3 加载完成")
    return net

def detect_objects(image, net):
    """使用 YOLOv3 检测人人"""
    height, width = image.shape[:2]

    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)

    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)

    boxes = []
    confidences = []

    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)

            # COCO 数据集中 person 的 ID 是 0
            if class_id != 0:
                continue

            confidence = scores[class_id]

            if confidence > 0.4:
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
            detections.append([x, y, w, h])

    return detections

def simple_tracker(detections, prev_tracks, max_distance=100):
    """
    简单的追踪器（基于 IoU 匹配）
    """
    tracks = []

    if len(prev_tracks) == 0:
        for i, det in enumerate(detections):
            tracks.append({'id': i+1, 'bbox': det, 'age': 0})
        return tracks

    used_dets = set()
    for track in prev_tracks:
        best_iou = 0
        best_det_idx = -1

        for j, det in enumerate(detections):
            if j in used_dets:
                continue

            x1 = max(track['bbox'][0], det[0])
            y1 = max(track['bbox'][1], det[1])
            x2 = min(track['bbox'][0]+track['bbox'][2], det[0]+det[2])
            y2 = min(track['bbox'][1]+track['bbox'][3], det[1]+det[3])

            inter = max(0, x2-x1) * max(0, y2-y1)
            union = track['bbox'][2]*track['bbox'][3] + det[2]*det[3] - inter

            if union > 0:
                iou = inter / union
                if iou > 0.3:
                    best_iou = iou
                    best_det_idx = j

        if best_iou > 0.3:
            track['bbox'] = detections[best_det_idx]
            track['age'] = 0
            tracks.append(track)
            used_dets.add(best_det_idx)
        elif track['age'] < 5:
            track['age'] += 1
            tracks.append(track)

    for i, det in enumerate(detections):
        if i not in used_dets:
            new_id = max([t['id'] for t in tracks], default=0) + 1
            tracks.append({'id': new_id, 'bbox': det, 'age': 0})

    return tracks

def run_tracking(seq_dir, output_path=None, max_frames=300):
    """运行追踪"""
    print(f"处理序列：{seq_dir}")

    net = load_yolo_detector()

    # 获取图像文件列表
    img_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
    total_frames = min(len(img_files), max_frames)

    print(f"找到 {len(img_files)} 张图像，处理前 {total_frames} 张")

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        first_img = cv2.imread(os.path.join(seq_dir, img_files[0]))
        height, width = first_img.shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, 15, (width, height))

    tracks = []

    for frame_idx, img_file in enumerate(img_files[:total_frames]):
        img_path = os.path.join(seq_dir, img_file)
        frame = cv2.imread(img_path)

        # 每 3 帧检测一次
        if frame_idx % 3 == 0:
            detections = detect_objects(frame, net)
            tracks = simple_tracker(detections, tracks)

        # 绘制结果
        for track in tracks:
            if track['age'] >= 5:
                continue

            x, y, w, h = map(int, track['bbox'])

            color = (int(hash(str(track['id'])) % 255),
                    int(hash(str(track['id'])[::-1]) % 255),
                    int(hash(str(track['id'])*2) % 255))

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID: {track['id']}", (x, y-10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        info = f"Frame: {frame_idx+1}/{total_frames}, Tracks: {len([t for t in tracks if t['age']<5])}"
        cv2.putText(frame, info, (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        if output_path:
            out.write(frame)

        if (frame_idx + 1) % 50 == 0:
            progress = 100 * (frame_idx + 1) / total_frames
            print(f"进度：{progress:.1f}% ({frame_idx+1}/{total_frames})")

    if output_path:
        out.release()

    print(f"完成！输出：{output_path}")

if __name__ == "__main__":
    output_video = "/app/output_tracking.mp4"
    run_tracking(MOT17_SEQ, output_video, max_frames=300)
