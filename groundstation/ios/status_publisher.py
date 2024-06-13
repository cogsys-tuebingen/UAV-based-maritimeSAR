import socket
import config
from data_structures import GroundStationStatusPacket
import time
import pickle


def status_publisher_thread(root):
    tcp_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_s.connect((config.ARM_DEVICE_SYSTEM_IP, config.ARM_DEVICE_CONTROL_PORT))
    while root.running:
        status_packet = GroundStationStatusPacket(time.time_ns(), root.get_custom_rois_copy())
        # print(f"! Send status packet: {status_packet}")
        coded_status_packet = pickle.dumps(status_packet)
        # tcp_s.send(bytes(coded_status_packet,encoding="utf-8"))
        tcp_s.send(coded_status_packet)
        time.sleep(0.1)
