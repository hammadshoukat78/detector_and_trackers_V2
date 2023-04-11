import torch
import cv2


class YoloDetector:
    def __init__(self, conf_thold=0.50, device="cpu",
                 weights_path="Utils/ObjectDetectors/Models/YoloV7/yolov7.pt",
                 expected_objs=None):
        # taking the image file path, confidence level and GPU or CPU selection
        if expected_objs is None:
            expected_objs = [1, 2, 3]
        self._model = torch.hub.load("WongKinYiu/yolov7", "custom", f"{weights_path}", trust_repo=True)
        self._model.conf = conf_thold  # NMS confidence threshold
        self._model.classes = expected_objs  # (optional list) filter by class
        self._model.to(device)  # specifying device type
        self._names = self._model.names

    @property
    def names(self):
        return self._names

    def process_image(self, image):
        results = self._model(image)
        predictions = []  # final list to return all detections
        detection = results.pandas().xyxy[0]
        # print(detection)
        for i in range(len(detection)):
            # getting bbox and class name one by one
            class_name = detection["name"]
            confidence_score = detection["confidence"]
            id = detection["class"]
            xmin = detection["xmin"]
            ymin = detection["ymin"]
            xmax = detection["xmax"]
            ymax = detection["ymax"]
            # parallely appending the values in list using dictionary
            predictions.append(
                {'bbox': [int(xmin[i]), int(ymin[i]), int(xmax[i]), int(ymax[i])], 'class_id': id[i],
                 "confidence": confidence_score[i], "class_name": class_name[i]})

        return predictions

