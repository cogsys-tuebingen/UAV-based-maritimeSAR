class DetRes():
    def __init__(self, scores, labels, boxes, scale):
        self.scores = scores
        self.labels = labels
        self.boxes = boxes
        self.scale = scale