import cv2
import numpy as np

def post_process_yolo(detections, img_w, img_h, conf_thres=0.5, iou_thres=0.6):
    boxes, scores, classes = [], [], []
    for det in detections:
        conf = det[4] * max(det[5:])  # Objectness * max class score
        if conf > conf_thres:
            x_center, y_center, w, h = det[0:4]
            x_min = int((x_center - w / 2) * img_w)
            y_min = int((y_center - h / 2) * img_h)
            x_max = int((x_center + w / 2) * img_w)
            y_max = int((y_center + h / 2) * img_h)
            boxes.append([x_min, y_min, x_max, y_max])
            scores.append(conf)
            classes.append(np.argmax(det[5:]))
    
    # Apply NMS
    indices = cv2.dnn.NMSBoxes(boxes, scores, conf_thres, iou_thres)
    if len(indices) > 0:
        indices = indices.flatten()
        return [boxes[i] for i in indices], [scores[i] for i in indices], [classes[i] for i in indices]
    else:
        return [], [], []
