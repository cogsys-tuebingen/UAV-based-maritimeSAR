import numpy as np
import time
import socket
import pickle

import config
import main


def metadata_retrieve_thread_demo(root):
    while root.running:
        a = np.random.randint(0, 500, (3, 4), dtype=int)
        a[:, 2:] -= a[:, :2]

        root.set_rois(a)
        time.sleep(3)


def metadata_retrieve_thread_tcp(root: main):
    print(f'# Connect to meta data server {config.ARM_DEVICE_SYSTEM_IP}:{config.ARM_DEVICE_METADATA_PORT}')

    while root.running:
        try:
            tcp_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            tcp_s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            tcp_s.connect((config.ARM_DEVICE_SYSTEM_IP, config.ARM_DEVICE_METADATA_PORT))
            tcp_s.settimeout(5)
            while root.running:
                try:
                    msg = tcp_s.recv(4096)
                    meta_data = pickle.loads(msg)
                    if config.USE_ARRIVAL_TIMESTAMP_FOR_SYNC:
                        timestamp = time.time_ns() // 1e7 + config.VIDEO_STREAM_DELAY
                    else:
                        timestamp = int(meta_data.timestamp * 1e-6)
                    root.update_state(meta_error=False, meta_connected=True)
                    root.synchronizer.add_rois_for_frame(timestamp, np.array(meta_data.rois).astype(int))
                except socket.timeout:
                    pass
                except pickle.UnpicklingError as e:
                    root.update_state(meta_error=True)
                    pass
                except EOFError as e:
                    root.update_state(meta_error=True)
                    pass
                except UnicodeError as e:
                    print(f"Ignore MetaData error: {e}")
                except KeyError as e:
                    print(f"Ignore MetaData error: {e}")
                except UnicodeDecodeError as e:
                    root.update_state(meta_error=True)

        except ConnectionRefusedError:
            root.update_state(meta_error=True)
            print("! Could not connect to metadata server. Retry in a second..")
            time.sleep(1)
