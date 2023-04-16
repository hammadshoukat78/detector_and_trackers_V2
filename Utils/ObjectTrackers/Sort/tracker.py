import numpy as np
import sys
import cv2


from Utils.ObjectDetectors.YoloV7.YoloDetector import YoloDetector
from Utils.ObjectTrackers.Sort.Helper.sort import Sort
from Utils.ObjectTrackers.Sort.Helper.utils import *


class SortTracker:
    def __init__(self,
                 source="C:/Users/hammad/Desktop/detectors_trackers_v2/trafic.mp4",
                 yolo_weights="Utils/ObjectDetectors/Models/YoloV7/yolov7.pt",
                 device="cpu",
                 confidence_thold=0.5,
                 exp_obj=[0, 1, 2]
                 ):
        self._source = source
        # initializing detector
        self._yolo_detector = YoloDetector(weights_path=yolo_weights,
                                           device=device, expected_objs=exp_obj, conf_thold=confidence_thold)
        # initializing Sort Tracker
        self._tracker = Sort(max_age=3)

    def process_frame(self, frame):
        detection = self._yolo_detector.process_image(image=frame)
        detections = dict_to_list(detection)
        tracker_results = self._tracker.update(dets=np.array(detections))
        output = []
        for item in tracker_results:
            bbox = item[:4]
            det_class = item[4]
            track_id = item[-1]
            output.append({'bbox': bbox, 'tracking_id': track_id, 'object_name': det_class})
        return output

    def process_video(self):
        cap = cv2.VideoCapture(self._source)
        assert cap.isOpened(), f'Failed to open {self._source}'
        while cap.isOpened():
            is_frame, frame = cap.read()
            if is_frame:
                tracker_results = self.process_frame(frame)
                frame = plot_bbox(tracker_results, frame, self._yolo_detector.names)
                cv2.imshow("tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()


