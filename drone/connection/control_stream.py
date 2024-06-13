import pickle
import socket

import global_vars
import config


def control_stream_loop(verbose):
    tcp_s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    tcp_s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    print('Starting up on {} port {} via TCP'.format(
        config.LOCAL_IP, config.CONTROL_PORT))
    tcp_s.bind((config.LOCAL_IP, config.CONTROL_PORT))
    tcp_s.listen(10)
    tcp_s.settimeout(1)

    while global_vars.is_running:
        try:
            tcp_clientsocket, tcp_address = tcp_s.accept()
            print("# Status data socket: client connected")

            while global_vars.is_running:
                try:
                    status_msg = tcp_clientsocket.recv(1024)
                    status_msg = pickle.loads(status_msg)
                    global_vars.rois = status_msg.custom_rois

                    if verbose:
                        print(f'New ROI proposal received from Client: {global_vars.rois}')
                except (pickle.UnpicklingError, EOFError, UnicodeDecodeError, ModuleNotFoundError) as e:
                    print(f"? Ignore invalid status packet of groundstation: {e}")

        except (BrokenPipeError, ConnectionRefusedError) as e:
            print(f"! Status data socket connection error: {e}. Restart..")
            time.sleep(1)
        except socket.timeout:
            pass
        except Exception as e:
            print(f"! StatusData: {e}")
            global_vars.cancel_signal(e)
    print("# Status data socket stopped.")

