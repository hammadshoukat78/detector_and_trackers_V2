import numpy as np
import sys
import cv2

sys.path.append("Utils/ObjectTrackers/ByteTrack")
from Utils.ObjectDetectors.YoloV7.YoloDetector import YoloDetector
from Utils.ObjectTrackers.ByteTrack.Helper.byte_tracker import BYTETracker
from Utils.ObjectTrackers.ByteTrack.Helper.utils import *


class ByteTracker:
    def __init__(self,
                 source="C:/Users/hammad/Desktop/detectors_trackers_v2/trafic.mp4",
                 yolo_weights="Utils/ObjectDetectors/Models/YoloV7/yolov7.pt",
                 device="cpu",
                 confidence_thold=0.5,
                 exp_obj=[0, 1, 2]):
        self._source = source
        # initializing bytetrack
        self._tracker = BYTETracker(track_thresh=confidence_thold)
        # initializing detector
        self._yolo_detector = YoloDetector(weights_path=yolo_weights,
                                           device=device, expected_objs=exp_obj, conf_thold=confidence_thold)

    def process_image(self, frame):
        detections = self._yolo_detector.process_image(image=frame)
        detection_results = convert_to_list_of_tuple(detections)
        # passing the list of tuples into trackers update function
        tracker_results = self._tracker.update(detections=detection_results)

        return convert_to_list_of_dict(tracker_results)

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

