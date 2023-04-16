import cv2


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


def convert_to_list_of_dict(detections):
    outputs = []
    for track in detections:
        track_id = track.track_id
        bbox = track.tlbr
        bbox = bbox.tolist()
        bbox = [int(x) for x in bbox]
        det_class = track.det_class
        outputs.append({'bbox': bbox, 'tracking_id': track_id, 'object_name': det_class})

    return outputs


def convert_to_list_of_tuple(detections):
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
    return detection_results
