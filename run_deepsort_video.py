#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
DeepSORT 视频追踪脚本 - 支持任意视频文件
"""

import sys
import os
import cv2
import numpy as np

YOLO_WEIGHTS = "/app/deep_sort_model/yolov3.weights"
YOLO_CFG = "/app/deep_sort_yolov3/yolov3.cfg"

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
            if class_id != 0:  # 只检测人
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
            if j in used: continue
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

def run_tracking(video_path, output_path="/app/output_tracking.mp4", max_frames=0):
    print(f"处理视频：{video_path}")
    net = load_yolo_detector()
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"无法打开视频：{video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"视频：{width}x{height}, {fps}fps, {total}帧")

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    tracks, frame_count = [], 0
    max_frames = max_frames if max_frames > 0 else total

    while frame_count < max_frames:
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1

        if frame_count % 3 == 0:
            dets = detect_objects(frame, net)
            tracks = simple_tracker(dets, tracks)

        for track in tracks:
            if track['age'] >= 5: continue
            x, y, w, h = map(int, track['bbox'])
            color = (hash(str(track['id'])) % 255, hash(str(track['id'])*2) % 255, hash(str(track['id'])*3) % 255)
            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, f"ID:{track['id']}", (x, y-8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        cv2.putText(frame, f"Frame:{frame_count}/{max_frames} Tracks:{len([t for t in tracks if t['age']<5])}",
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        out.write(frame)

        if frame_count % 50 == 0:
            print(f"进度：{100*frame_count/max_frames:.1f}% ({frame_count}/{max_frames})")

    cap.release()
    out.release()
    print(f"完成！输出：{output_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python run_deepsort_video.py <视频路径> [最大帧数]")
        sys.exit(1)

    video = sys.argv[1]
    max_f = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    run_tracking(video, "/app/output_tracking.mp4", max_f)
