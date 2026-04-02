#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSORT MOT17 序列追踪 - 指定任意序列
用法：python run_mot17.py MOT17-02-FRCNN [最大帧数]
"""

import sys
import os
import cv2
import numpy as np

YOLO_WEIGHTS = "/app/deep_sort_model/yolov3.weights"
YOLO_CFG = "/app/deep_sort_yolov3/yolov3.cfg"
MOT17_ROOT = "/mnt/MOT17/train"

def load_yolo_detector():
    print("加载 YOLOv3 检测器...")
    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
    print("YOLOv3 加载完成")
    return net

def detect_objects(image, net):
    height, width = image.shape[:2]
    blob = cv2.dnn.blobFromImage(image, 1/255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    outputs = net.forward(output_layers)
    boxes, confidences = [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            if class_id != 0:
                continue
            confidence = scores[class_id]
            if confidence > 0.4:
                cx, cy = int(detection[0] * width), int(detection[1] * height)
                w, h = int(detection[2] * width), int(detection[3] * height)
                boxes.append([cx - w//2, cy - h//2, w, h])
                confidences.append(float(confidence))
    indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    return [boxes[i] for i in indices.flatten()] if len(indices) > 0 else []

def simple_tracker(detections, prev_tracks):
    if not prev_tracks:
        return [{'id': i+1, 'bbox': det, 'age': 0} for i, det in enumerate(detections)]
    tracks, used = [], set()
    for track in prev_tracks:
        best_iou, best_idx = 0, -1
        for j, det in enumerate(detections):
            if j in used:
                continue
            x1 = max(track['bbox'][0], det[0])
            y1 = max(track['bbox'][1], det[1])
            x2 = min(track['bbox'][0]+track['bbox'][2], det[0]+det[2])
            y2 = min(track['bbox'][1]+track['bbox'][3], det[1]+det[3])
            inter = max(0, x2-x1) * max(0, y2-y1)
            union = track['bbox'][2]*track['bbox'][3] + det[2]*det[3] - inter
            if union > 0:
                iou = inter / union
                if iou > best_iou:
                    best_iou, best_idx = iou, j
        if best_iou > 0.3:
            track['bbox'], track['age'] = detections[best_idx], 0
            tracks.append(track)
            used.add(best_idx)
        elif track['age'] < 5:
            track['age'] += 1
            tracks.append(track)
    for i, det in enumerate(detections):
        if i not in used:
            tracks.append({'id': max([t['id'] for t in tracks], default=0)+1, 'bbox': det, 'age': 0})
    return tracks

def run_tracking(seq_name, output_dir, max_frames=300):
    seq_dir = f"{MOT17_ROOT}/{seq_name}/img1"
    if not os.path.exists(seq_dir):
        print(f"序列不存在：{seq_dir}")
        return None

    img_files = sorted([f for f in os.listdir(seq_dir) if f.endswith('.jpg')])
    if not img_files:
        print(f"未找到图片：{seq_dir}")
        return None

    total = min(len(img_files), max_frames)
    print(f"序列：{seq_name}")
    print(f"图片数：{len(img_files)}, 处理前 {total} 张")

    net = load_yolo_detector()
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"{seq_name}_output.mp4")

    first_img = cv2.imread(os.path.join(seq_dir, img_files[0]))
    height, width = first_img.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, 15, (width, height))

    tracks = []
    for i, img_file in enumerate(img_files[:total]):
        frame = cv2.imread(os.path.join(seq_dir, img_file))
        if i % 3 == 0:
            dets = detect_objects(frame, net)
            tracks = simple_tracker(dets, tracks)
        for track in tracks:
            if track['age'] >= 5:
                continue
            x, y, w, h = map(int, track['bbox'])
            color = (hash(str(track['id'])) % 255, hash(str(track['id'])*2) % 255, hash(str(track['id'])*3) % 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID:{track['id']}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        cv2.putText(frame, f"{i+1}/{total}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out.write(frame)
        if (i+1) % 50 == 0:
            print(f"进度：{100*(i+1)/total:.1f}%")

    out.release()
    print(f"完成：{output_path}")
    return output_path

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python run_mot17.py <序列名> [最大帧数] [输出目录]")
        print("序列名示例：MOT17-02-FRCNN, MOT17-04-DPM, MOT17-10-SDP")
        sys.exit(1)
    seq = sys.argv[1]
    max_f = int(sys.argv[2]) if len(sys.argv) > 2 else 300
    out_dir = sys.argv[3] if len(sys.argv) > 3 else "/app/results"
    run_tracking(seq, out_dir, max_f)
