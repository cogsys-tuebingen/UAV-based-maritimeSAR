class MetaDataStructure:
    def __init__(self, frame_id: int, timestamp, rois: list):
        self.frame_id = frame_id
        self.timestamp = timestamp
        self.rois = rois

    def update_frame_id(self, frame_id):
        self.frame_id = frame_id

    def update_timestamp(self, timestamp):
        self.timestamp = timestamp