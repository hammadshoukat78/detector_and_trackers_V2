import cv2
import torch
import sys
import numpy as np

from Utils.ObjectDetectors.YoloV7.YoloDetector import YoloDetector
from Utils.ObjectTrackers.DeepSort.Helper.deep_sort.deep_sort import DeepSort
from Utils.ObjectTrackers.DeepSort.Helper.deep_sort.utils.parser import get_config


class DeepSortTracker():
    def __init__(self, source="C:/Users/hammad/Desktop/detectors_trackers_v2/trafic.mp4",
                 config_deepsort_path="Utils\\ObjectTrackers\\DeepSort\\Helper\\deep_sort\\configs\\deep_sort.yaml",
                 yolo_weights="Utils/ObjectDetectors/Models/YoloV7/yolov7.pt",
                 device="cpu",
                 confidence_thold=0.5,
                 exp_obj=[0, 1, 2]
                 ):
        # initialize deepsort
        cfg = get_config()
        cfg.merge_from_file(config_deepsort_path)
        self._deepsort_tracker = DeepSort('osnet_x0_25',
                                          max_dist=cfg.DEEPSORT.MAX_DIST,
                                          max_iou_distance=cfg.DEEPSORT.MAX_IOU_DISTANCE,
                                          max_age=cfg.DEEPSORT.MAX_AGE, n_init=cfg.DEEPSORT.N_INIT,
                                          nn_budget=cfg.DEEPSORT.NN_BUDGET,
                                          use_cuda=True)
        # initialize yolo detector
        self._yolo_detector = YoloDetector(weights_path=yolo_weights,
                                           device=device, expected_objs=exp_obj, conf_thold=confidence_thold)
        self._names = self._yolo_detector.names
        self._source = source

    def process_image(self, frame):
        detections = self._yolo_detector.process_image(image=frame)
        detections_per_frame = []
        # receiving a list of dictionaries from detector [{bbox, conf, class_id}]
        for item in detections:
            bbox = item['bbox']
            confidence = item['confidence']
            obj_class = item['class_id']
            detections_per_frame.append(bbox + [confidence, obj_class])

        det_in_tensor = [torch.tensor(detections_per_frame)]
        for i, det in enumerate(det_in_tensor):
            # print(det)
            if det is not None and len(det):
                det = det[:, :6]
                xywhs = xyxy2xywh(det[:, 0:4])
                confs = det[:, 4]
                clss = det[:, 5]
                # updating all the detections into deepsort after all the processing
                outputs = self._deepsort_tracker.update(xywhs.cpu(), confs.cpu(), clss.cpu(), frame)
                predictions = []
                for i in range(len(outputs)):
                    bbox = outputs[i][0:4]
                    bbox = bbox.tolist()
                    bbox = [int(x) for x in bbox]
                    tracking_id = outputs[i][4]
                    obj_id = outputs[i][-1]
                    class_names = self._names
                    obj_name = class_names[obj_id]
                    # getting a List of dictionaries just like detector
                    predictions.append({'bbox': bbox, 'tracking_id': tracking_id, 'object_name': obj_name})

                return predictions

    def process_video(self):
        cap = cv2.VideoCapture(self._source)
        assert cap.isOpened(), f'Failed to open {self._source}'
        while cap.isOpened():
            is_frame, frame = cap.read()
            if is_frame:
                tracker_results = self.process_image(frame)
                frame = plot_bbox(tracker_results, frame)

                cv2.imshow("tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()


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