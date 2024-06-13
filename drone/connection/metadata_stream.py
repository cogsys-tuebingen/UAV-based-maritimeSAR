import pickle
import socket
import queue

import global_vars
import config


def meta_data_stream_loop(coordinator, protocol='tcp'):
    _meta_data_transmission_loop_tcp(coordinator)


def _meta_data_transmission_loop_tcp(coordinator):
    print(f"# Open meta data socket ({config.LOCAL_IP}:{config.METADATA_PORT})")
    tcp_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    tcp_s.bind((config.LOCAL_IP, config.METADATA_PORT))
    tcp_s.settimeout(5)
    tcp_s.listen(10)

    while global_vars.is_running:
        try:
            tcp_clientsocket, tcp_address = tcp_s.accept()
            print("# Meta data socket: client connected")

            while not coordinator.meta_data_queue.empty():
                coordinator.meta_data_queue.get()

            while global_vars.is_running:
                try:
                    meta_data = coordinator.meta_data_queue.get(timeout=2)
                    msg = pickle.dumps(meta_data)
                    tcp_clientsocket.send(msg)
                except queue.Empty:
                    pass
                except ConnectionResetError:
                    print("Metadatastream: Connection reset by peer, waiting for new connections")
        except socket.timeout:
            pass
        except BrokenPipeError:
            print("! Meta data socket: Broke pipe. Restart..")
        except Exception as e:
            print(f"! MetaData: {e}")
            global_vars.cancel_signal(e)
    print("# Meta data socket stopped.")

