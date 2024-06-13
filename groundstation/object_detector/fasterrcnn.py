import numpy as np
import torch
import torchvision
from .interfaces import TrainedModelInterface, Detection
from faster_rcnn.model import fasterrcnn_resnet_fpn


class FasterRCNN(TrainedModelInterface):
    def __init__(self, ckpt_path: str, device_id: str = 'cuda:0', num_classes=91, backbone='resnet50'):
        super(FasterRCNN, self).__init__()
        self.device = torch.device(device_id)
        self.model = fasterrcnn_resnet_fpn(pretrained=False, progress=True,
                                           num_classes=num_classes, backbone_arch=backbone,
                                           pretrained_backbone=True,
                                           box_nms_thresh=0.3)
        self.model.to(self.device)
        self.model.eval()

        if ckpt_path is not None:
            state_dict = torch.load(ckpt_path).state_dict()
            self.model.load_state_dict(state_dict)

    def inference(self, roi_images: [np.array], confidence_threshold: float) -> [[Detection]]:
        if len(roi_images) == 0:
            return []

        roi_images = [img / 255 if np.amax(img) > 1. else img for img in roi_images]
        tensor_list = [torch.from_numpy(a) for a in roi_images]
        max_dim = [max([t.size(j) for t in tensor_list]) for j in range(3)]
        for k, roi in enumerate(tensor_list):
            background_tensor = torch.zeros(max_dim)
            background_tensor[:roi.size(0), :roi.size(1), :roi.size(2)] = roi
            tensor_list[k] = background_tensor.permute([2, 0, 1]).unsqueeze(0)

        roi_tensor = torch.cat(tensor_list).to(self.device)

        with torch.no_grad():
            predictions = self.model(roi_tensor)
        return_array = []
        for roi in predictions:
            boxes = roi['boxes']
            boxes[:, 2:] -= boxes[:, :2]
            return_array.append(torch.cat([boxes,
                                           roi['labels'].unsqueeze(1),
                                           roi['scores'].unsqueeze(1)],
                                          dim=1))

        return_array = [FasterRCNN.convert_to_detection(p, confidence_threshold) for p in return_array]

        return return_array

    @staticmethod
    def convert_to_detection(detection: torch.tensor, confidence_threshold: float = 0):
        """
        :param confidence_threshold:
        :param detection: Is expected to be of shape N times 6.
        :return:
        """
        detectionslist = []
        for line in detection:
            x, y, w, h, class_id, confidence = line
            if confidence < confidence_threshold:
                continue
            detectionslist.append(
                Detection(int(x.item()), int(y.item()), int(w.item()), int(h.item()), int(class_id.item()),
                          confidence.item()))
        return detectionslist
