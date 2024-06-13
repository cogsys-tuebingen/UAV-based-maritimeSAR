import time
import pickle
import socket
import os

import global_vars
import config


def dataframe_store_loop(max_fps, verbose, path):
    if verbose:
        start = time.time()
        frmcntr = 0

    os.makedirs(path, exist_ok=True)

    print(f'Starting storage thread and save packet dumps into {path}')

    while global_vars.is_running:
        timer1 = time.time()

        global_vars.det_lock.acquire()
        data_frame = global_vars.last_data_frame
        global_vars.det_lock.release()

        file_name = f"packet_{time.time()}.pt"
        pickle.dump(data_frame, open(os.path.join(path, file_name)))

        timer2 = time.time()
        time_diff = (1 / max_fps) - (timer2 - timer1)
        if time_diff > 0:
            time.sleep(time_diff)
