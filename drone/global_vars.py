import threading

last_cv_frame = None
last_data_frame = None

hdlr_lock = threading.Lock()
det_lock = threading.Lock()

handler_event = threading.Event()
new_dataframe_event = threading.Event()

rois = []

# Constants
HEADERSIZE = 20

is_running = True
is_running_detection = True # needed to stop detection thread last to avoid pycuda abortion
thread_holder = []
loop = None # Loop for rtsp_stream

def cancel_signal(*args):
    print(f"Cancel.")
    global is_running
    loop.quit()
    is_running = False
