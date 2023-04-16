import cv2


def plot_bbox(outputs, frame, names):
    for item in outputs:
        bbox = item['bbox']
        tracking_id = int(item['tracking_id'])
        object_name = names[int(item['object_name'])]
        # Draw bounding box
        cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (0, 0, 225), 2)
        # Put text on the box
        text = f"{tracking_id} {object_name}"
        cv2.putText(frame, text, (int(bbox[0]), int(bbox[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 225), 2)
    return frame


def dict_to_list(detection):
    dets = []
    for i in detection:
        # getting bbox and class name one by one
        bbox = i['bbox']
        class_id = i['class_id']
        confidence = i['confidence']
        dets.append([bbox[0], bbox[1], bbox[2], bbox[3], confidence, class_id])
    return dets