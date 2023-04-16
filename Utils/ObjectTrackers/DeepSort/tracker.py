import cv2
import torch
import sys
import numpy as np

from Utils.ObjectDetectors.YoloV7.YoloDetector import YoloDetector
from Utils.ObjectTrackers.DeepSort.Helper.deep_sort.deep_sort import DeepSort
from Utils.ObjectTrackers.DeepSort.Helper.deep_sort.utils.parser import get_config
from Utils.ObjectTrackers.DeepSort.Helper.utils import *


class DeepSortTracker:
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
        self._tracker = DeepSort('osnet_x0_25',
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

    def process_frame(self, frame):
        detections = self._yolo_detector.process_image(image=frame)
        bbox, confs, clss = detections_for_tracker(detections)
        # updating all the detections into deepsort after all the processing
        outputs = self._tracker.update(bbox, confs, clss, frame)
        return convert_to_list_of_dict(outputs, self._yolo_detector.names)

    def process_video(self):
        cap = cv2.VideoCapture(self._source)
        assert cap.isOpened(), f'Failed to open {self._source}'
        while cap.isOpened():
            is_frame, frame = cap.read()
            if is_frame:
                tracker_results = self.process_frame(frame)
                frame = plot_bbox(tracker_results, frame)
                cv2.imshow("tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()