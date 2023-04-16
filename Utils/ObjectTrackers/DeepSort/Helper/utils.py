import numpy as np
import cv2
import torch


def plot_bbox(outputs, frame):
    for item in outputs:
        bbox = item['bbox']
        tracking_id = item['tracking_id']
        object_name = item['object_name']

        # Draw bounding box
        cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 225), 2)

        # Put text on the box
        text = f"{tracking_id} {object_name}"
        cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)
    return frame


def xyxy2xywh(x):
    # Convert nx4 boxes from [x1, y1, x2, y2] to [x, y, w, h] where xy1=top-left, xy2=bottom-right
    y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
    y[..., 0] = (x[..., 0] + x[..., 2]) / 2  # x center
    y[..., 1] = (x[..., 1] + x[..., 3]) / 2  # y center
    y[..., 2] = x[..., 2] - x[..., 0]  # width
    y[..., 3] = x[..., 3] - x[..., 1]  # height
    return y


def convert_to_list_of_dict(det, class_names):
    predictions = []
    for i in range(len(det)):
        bbox = det[i][0:4]
        bbox = bbox.tolist()
        bbox = [int(x) for x in bbox]
        tracking_id = det[i][4]
        obj_id = det[i][-1]
        obj_name = class_names[obj_id]
        # getting a List of dictionaries just like detector
        predictions.append({'bbox': bbox, 'tracking_id': tracking_id, 'object_name': obj_name})

    return predictions


def detections_for_tracker(detections):
    detections_per_frame = []
    # receiving a list of dictionaries from detector [{bbox, conf, class_id}]
    for item in detections:
        bbox = item['bbox']
        confidence = item['confidence']
        obj_class = item['class_id']
        detections_per_frame.append(bbox + [confidence, obj_class])

    det_in_tensor = [torch.tensor(detections_per_frame)]
    for i, det in enumerate(det_in_tensor):
        # print(f"inside det_in_tensor: i {i}, det {det}")
        if det is not None and len(det):
            det = det[:, :6]
            xywhs = xyxy2xywh(det[:, 0:4])
            confs = det[:, 4]
            clss = det[:, 5]
            return xywhs.cpu(), confs.cpu(), clss.cpu()