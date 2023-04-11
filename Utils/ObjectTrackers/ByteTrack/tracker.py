import numpy as np
import sys
import cv2

sys.path.append("Utils/ObjectTrackers/ByteTrack")
from Utils.ObjectDetectors.YoloV7.YoloDetector import YoloDetector
from Helper.byte_tracker import BYTETracker


class ByteTracker:
    def __init__(self,
                 source="C:/Users/hammad/Desktop/detectors_trackers_v2/trafic.mp4",
                 yolo_weights="Utils/ObjectDetectors/Models/YoloV7/yolov7.pt",
                 device="cpu",
                 confidence_thold=0.5,
                 exp_obj=[0, 1, 2]):
        self._source = source
        # initializing bytetrack
        self._byte_tracker = BYTETracker(track_thresh=confidence_thold)
        # initializing detector
        self._yolo_detector = YoloDetector(weights_path=yolo_weights,
                                           device=device, expected_objs=exp_obj, conf_thold=confidence_thold)

    def process_image(self, frame):
        detections = self._yolo_detector.process_image(image=frame)
        # list of dict are coming here resolve into list of tuples
        for bbox in detections:
            # Get the [xmin, ymin, xmax, ymax] coordinates
            bbox_xyxy = bbox['bbox']
            left = bbox_xyxy[0]
            top = bbox_xyxy[1]
            width = bbox_xyxy[2] - bbox_xyxy[0]
            height = bbox_xyxy[3] - bbox_xyxy[1]
            # Update the bbox coordinates in the dictionary
            bbox['bbox'] = [left, top, width, height]

        detection_results = [(item['bbox'], item['confidence'], item['class_name']) for item in detections]
        # passing the list of tuples into trackers update function
        tracker_results = self._byte_tracker.update(detections=detection_results)
        outputs = []
        for track in tracker_results:
            track_id = track.track_id
            bbox = track.tlbr
            bbox = bbox.tolist()
            bbox = [int(x) for x in bbox]
            det_class = track.det_class
            outputs.append({'bbox': bbox, 'tracking_id': track_id, 'object_name': det_class})

        return outputs

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
