import time
from threading import Thread

import global_vars


def capture_genicam(heigth, width, verbose):  # doesn't work on Jetson !
    raise RuntimeError("GenTL Producer not compatible with the ARM64 architecture")
    if verbose:
        start = time.time()
        frmcntr = 0

    h = Harvester()
    h.add_file('/opt/mvIMPACT_Acquire/lib/x86_64/mvGenTLProducer.cti')  # individual CTI filepath
    print(h.files)
    h.update()
    print(h.device_info_list)

    ia = h.create_image_acquirer(list_index=0)

    ia.remote_device.node_map.AcquisitionMode.value = 'Continuous'
    ia.remote_device.node_map.TriggerMode.value = 'On'
    ia.remote_device.node_map.TriggerSource.value = 'Software'
    # ia.remote_device.node_map.PixelFormat.value = 'RBG' # Point Grey Cam is only Mono !

    ia.start_acquisition(run_in_background=True)

    while global_vars.is_running:

        ia.remote_device.node_map.TriggerSoftware.execute()

        with ia.fetch_buffer() as buffer:

            payload = buffer.payload
            component = payload.components[0]
            width = component.width
            height = component.height
            data_format = component.data_format

            content = component.data.reshape(height, width)

            # cv2.imshow('Test Gen<i>Cam capture', content)
            # cv2.waitKey(1)

        if verbose:
            frmcntr += 1
            end = time.time()
            time_diff = end - start
            fps = frmcntr / time_diff
            print(f"Gen<i>CAM STREAM - fps mean: {fps}, frameID: {frmcntr}, time: {time_diff}")
            if frmcntr == 100:
                frmcntr = 0
                start = time.time()


def start_generic_genicam_thread(height, width, verbose):
    genicam_thread = Thread(target=capture_genicam, args=([width, height, verbose]))
    genicam_thread.start()
    global_vars.thread_holder.append(genicam_thread)