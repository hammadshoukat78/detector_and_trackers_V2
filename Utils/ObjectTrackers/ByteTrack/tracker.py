import numpy as np
import sys
import cv2

sys.path.append("Utils/ObjectTrackers/ByteTrack")
from Utils.ObjectDetectors.YoloV7.YoloDetector import YoloDetector
from Helper.byte_tracker import BYTETracker


class ByteTracker:
    def __init__(self):
        self._track_thresh = 0.5
        self._track_buffer = 30
        self._match_thresh = 0.8
        self._frame_rate = 25
        # initializing bytetrack
        self._tracker_obj = BYTETracker(
            track_thresh=self._track_thresh,
            track_buffer=self._track_buffer,
            match_thresh=self._match_thresh,
            mot20=False,
            frame_rate=25
        )
        # initializing detector
        self._detector_obj = YoloDetector()

    def process_image(self, frame):
        detections = self._detector_obj.process_image(image=frame)
        # list of dict are coming here resolve into list of tuples
        for bbox_dict in detections:
            # Get the [xmin, ymin, xmax, ymax] coordinates
            bbox_xyxy = bbox_dict['bbox']
            # Calculate the [left, top, width, height] coordinates
            left = bbox_xyxy[0]
            top = bbox_xyxy[1]
            width = bbox_xyxy[2] - bbox_xyxy[0]
            height = bbox_xyxy[3] - bbox_xyxy[1]
            # Update the bbox coordinates in the dictionary
            bbox_dict['bbox'] = [left, top, width, height]

        results = [(item['bbox'], item['confidence'], item['class_name']) for item in detections]
        # passing the list of tuples into trackers update function
        results = self._tracker_obj.update(detections=results)
        outputs = []
        for track in results:
            track_id = track.track_id
            bbox = track.tlbr
            bbox = bbox.tolist()
            bbox = [int(x) for x in bbox]
            det_class = track.det_class
            outputs.append({'bbox': bbox, 'tracking_id': track_id, 'object_name': det_class})

        return outputs

    def process_video(self, source):
        cap = cv2.VideoCapture(source)
        assert cap.isOpened(), f'Failed to open {source}'
        while cap.isOpened():
            is_frame, frame = cap.read()
            if is_frame:
                results = self.process_image(frame)
                print(results)
                for item in results:
                    bbox = item['bbox']
                    tracking_id = item['tracking_id']
                    object_name = item['object_name']

                    # Draw bounding box
                    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 0, 225), 2)

                    # Put text on the box
                    text = f"{tracking_id} {object_name}"
                    cv2.putText(frame, text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)
                cv2.imshow("tracking", frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        cap.release()
        cv2.destroyAllWindows()

