class MetaDataStructure:
    """
    Sent by the arm software
    """
    def __init__(self, frame_id: int, rois: []):
        self.frame_id = frame_id
        self.rois = rois

    def update_frame_id(self, frame_id):
        self.frame_id = frame_id