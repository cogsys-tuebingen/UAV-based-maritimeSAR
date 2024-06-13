import enum

class VIDEO_TRANSMISSION_ENUM(enum.Enum):
    RTSP = 'rtsp'
    UDP = 'udp'


CERTAINTY_THRESHOLD = 0.7
OBJECT_DETECTOR_PTH = 'resources/trained_model_august.pth'
OBJECT_DETECTOR_BACKBONE = 'resnext101_32x8d'

ARM_DEVICE_SYSTEM_IP = "192.168.0.2"
ARM_DEVICE_METADATA_PORT = 5005
ARM_DEVICE_CONTROL_PORT = 9999
ARM_DEVICE_VIDEO_IP = "192.168.0.2"
ARM_DEVICE_UDP_STREAM_PORT = [5000, 5001]

CLASSES = [
    {'supercategory': 'ignored', 'id': 0, 'name': 'ignored region'},
    {'supercategory': 'person', 'id': 1, 'name': 'swimmer'},
    {'supercategory': 'boat', 'id': 2, 'name': 'boat'},
    {'supercategory': 'boat', 'id': 3, 'name': 'jetski'},
    {'supercategory': 'object', 'id': 4, 'name': 'jetski'},
    {'supercategory': 'object', 'id': 5, 'name': 'buoy'}
]

NUM_CLASSES = 20

VIDEO_TRANSMISSION: VIDEO_TRANSMISSION_ENUM = VIDEO_TRANSMISSION_ENUM.RTSP
COMBINE_TWO_VIDEO_STREAMS = False
SYNC_PREDICTIONS = True
USE_ARRIVAL_TIMESTAMP_FOR_SYNC = False
VIDEO_STREAM_DELAY = 400

VISIBLE_CUDA_DEVICES = ['cuda:0']
