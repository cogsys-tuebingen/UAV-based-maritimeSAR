import time
import torch
from torchvision import transforms
import cv2
import os
import copy
import numpy as np
import pycuda.driver as cuda
import copy
import matplotlib.pyplot as plt

import global_vars
from util.data_frame import DataFrame


def get_rois(detected_rois, custom_rois):
    return detected_rois + custom_rois


def detection_loop(detector, detection_frequency, detector_size, verbose, saliency,
                   best_percent=0.1, enlarge_bb=1.4, detection_threshold=0.6, average_roi_frames=3):
    from .resnet import resnet18
    from .trt_resnet import TrtResnet
    from .TrtEffDet import TrtEffDet
    from .gradcam_mask import SaliencyMaskDropout
    from .gradcam import GradCAM, GradCAMpp

    frames_since_last_detection = 0

    if verbose:
        start = time.time()
        frmcntr = 0

    cuda.init()
    device = cuda.Device(0)
    device_ctx = device.make_context()

    if saliency:
        masker = SaliencyMaskDropout(keep_percent=best_percent, scale_map=False)
        saliencynet = TrtResnet(detector, detector_size, cuda_ctx=device_ctx)  # (960, 540)

        resnet = resnet18(pretrained=False)
        checkpoint = torch.load('trt_checkpoints/resnet18-f37072fd.pth')  # TODO: adapt to system
        resnet.load_my_state_dict(checkpoint)
        resnet.cuda().eval()
        gradcam_pp = GradCAMpp.from_config(arch=resnet, layer_name='layer4', model_type="resnet")
    else:
        effdet = TrtEffDet(detector, detector_size, cuda_ctx=device_ctx)

    current_dataframe_id = 1
    detected_rois = []
    detected_roi_scores = []




    dtr_times = []
    post_times = []

    if saliency:
        # smooth the rois prediction by averaging over two frames
        last_out = np.ones((average_roi_frames, detector_size[1], detector_size[0]), dtype=np.uint8) * 100
        replace_last_out_id = 0

    while global_vars.is_running_detection:
        global_vars.handler_event.wait(timeout=1)
        global_vars.handler_event.clear()

        global_vars.hdlr_lock.acquire()
        cv_frame = copy.deepcopy(global_vars.last_cv_frame)
        global_vars.last_cv_frame = None
        global_vars.hdlr_lock.release()
        global_vars.handler_event.clear()

        if cv_frame is None:
            continue
        dtr_start = 0
        dtr_end = 0
        post_start = 0
        post_end = 0

        if frames_since_last_detection >= detection_frequency:
            detected_rois = []
            detected_roi_scores = []
            if saliency:
                assert round(cv_frame.shape[1] / detector_size[0], 2) == round(cv_frame.shape[0] / detector_size[1], 2)
                scale = detector_size[0] / cv_frame.shape[1]
                np_img = cv2.resize(cv_frame, (detector_size[0], detector_size[1]),
                                    interpolation=cv2.INTER_CUBIC)
                                    

                # TODO:
                #cv2.imwrite(f'/tmp/heatmaps_orig/{frmcntr}_orig.png', np_img)
                                    
                                    
                torch_img = torch.from_numpy(np_img).cuda()  # resize half
                torch_img = torch_img.permute(2, 0, 1) / 255

                normed_torch_img = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]
                
                dtr_start = time.perf_counter()

                trt_out = saliencynet.detect(normed_torch_img.cpu().numpy())
                mask_pp, _ = gradcam_pp(normed_torch_img, trt_out=trt_out)
                heatmap_pp = (255 * mask_pp.squeeze()).type(torch.uint8)
                
                
                
                # TODO:
                disk_img = heatmap_pp.cpu().numpy()
                #print('\n\n\njojojo:', disk_img.shape)
                
                
                # TODO:
                #cv2.imwrite(f'/tmp/heatmaps/{frmcntr}.png', disk_img)
                #print('\n\n\ndfbglonjidrobgtrn:', disk_img.shape)
                
                # Create a figure and an axis
                #fig, ax = plt.subplots()

                # Apply the colormap
                # The 'jet' colormap goes from blue to red, passing through green, but you can experiment with others
                #cax = ax.imshow(disk_img, cmap='jet', interpolation='nearest')

                # Add a color bar to the side
                #fig.colorbar(cax)

                # Display the heatmap
                #plt.savefig(f'/tmp/heatmaps/{frmcntr}.png')
                #plt.show()
                
                
                
                dtr_end = time.perf_counter()
                post_start = time.perf_counter()

                out = masker(normed_torch_img, (heatmap_pp / 255).unsqueeze(0))
                
                
                
                out = (out[1].squeeze().cpu().numpy() * 255).astype(np.uint8)
                last_out[replace_last_out_id] = out
                #replace_last_out_id = (replace_last_out_id + 1) % average_roi_frames
                #top_mask = last_out.mean(0).astype(np.uint8)
                top_mask = out
                
                
                
                #disk_img = (255 * mask_pp.squeeze()).type(torch.uint8).cpu().numpy()
                # TODO:
                #cv2.imwrite(f'/tmp/heatmaps_binary/{frmcntr}.png', out)
                
                
                
                
                contours = cv2.findContours(top_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                contours = contours[0] if len(contours) == 2 else contours[1]
                for c in contours:
                    x1_, y1_, w_, h_ = cv2.boundingRect(c)

                    # filter the detections
                    if heatmap_pp[y1_:(y1_ + h_), x1_:(x1_ + w_)].max() < 100:
                        continue

                    x1_, y1_, w_, h_ = [int(p / scale) for p in cv2.boundingRect(c)]

                    d_w = -(w_ - (w_ * enlarge_bb))
                    d_h = -(h_ - (h_ * enlarge_bb))
                    x1 = max(0, x1_ - int(d_w / 2))
                    y1 = max(0, y1_ - int(d_h / 2))
                    x2 = x1_ + (w_ + int(d_w / 2))
                    y2 = y1_ + (h_ + int(d_h / 2))

                    detected_rois.append([x1, y1, (x2 - x1), (y2 - y1)])
                    detected_roi_scores = (1)
                    
                    post_end = time.perf_counter()
            else:
                scores, labels, boxes, scale = effdet.detect(cv_frame, [])
                boxes /= scale

                for score, label, box in zip(scores, labels, boxes):
                    if score > detection_threshold:
                        x1_, y1_, w_, h_ = box
                        detected_rois.append([x1_, y1_, w_, h_])
                        detected_roi_scores.append(score)

            frames_since_last_detection = 0

        frames_since_last_detection += 1

        data_frame = generate_dataframe_roi(current_dataframe_id, cv_frame, detected_rois,
                                            roi_scores=detected_roi_scores)
        current_dataframe_id += 1

        global_vars.det_lock.acquire()
        global_vars.last_data_frame = copy.deepcopy(data_frame)
        global_vars.det_lock.release()
        global_vars.new_dataframe_event.set()

        if verbose:
            frmcntr += 1
            end = time.time()
            time_diff = end - start
            fps = frmcntr / time_diff
            #print(f"DETECTOR - fps mean: {fps}, frameID: {frmcntr}, time: {time_diff}")
            if post_end-post_start >0 and post_end-post_start>0:
                #print(f"DETECTOR - time: {1000*(dtr_end-dtr_start)} ms, POSTPROCESSING - time: {1000*(post_end-post_start)} ms\nDETECTOR - fps: {1/(dtr_end-dtr_start)}, POSTPROCESSING - fps: {1/(post_end-post_start)}")
                post_times.append(post_end-post_start)
                dtr_times.append(dtr_end-dtr_start)
            if frmcntr == 100:
                #frmcntr = 0
                start = time.time()
                
    print("# Detection thread stopped.")
    print(f"DETECTOR - fps avg: {1/np.mean(dtr_times)}, POSTPROCESSING - fps avg: {1/np.mean(post_times)}")
    print(f"\n\nCLEANED\nDETECTOR - fps avg: {1/np.mean(dtr_times[15:])}, POSTPROCESSING - fps: {1/np.mean(post_times[15:])}")

    print(3*"\n")

    print(f"DETECTOR - time avg: {np.mean(dtr_times)}, POSTPROCESSING - time avg: {np.mean(post_times)}")
    print(f"\n\nCLEANED\nDETECTOR - time avg: {np.mean(dtr_times[15:])}, POSTPROCESSING - time: {np.mean(post_times[15:])}")

    print(3*"\n")
    print(f"avg size: {len(dtr_times)}")


def generate_dataframe_roi(id, image, rois, roi_scores):
    custom_rois = global_vars.rois
    data_frame = DataFrame(id=id, main_img=image, rois=rois, roi_scores=roi_scores, custom_rois=custom_rois)

    return data_frame


def detection_mockup():
    print("# Detection mockup")
    current_dataframe_id = 0

    while global_vars.is_running:
        global_vars.handler_event.wait(timeout=1)
        global_vars.handler_event.clear()

        global_vars.hdlr_lock.acquire()
        img = copy.deepcopy(global_vars.last_cv_frame)
        global_vars.last_cv_frame = None
        global_vars.hdlr_lock.release()
        global_vars.handler_event.clear()

        if img is None:
            continue

        global_vars.det_lock.acquire()
        pos = current_dataframe_id % 1200
        img[:, pos: pos + 50] = 0
        rois = [[pos, 0, 50, 720]]
        rois_score = [1]

        global_vars.last_data_frame = generate_dataframe_roi(current_dataframe_id, img, rois, roi_scores=rois_score)
        current_dataframe_id += 1
        global_vars.det_lock.release()
        global_vars.new_dataframe_event.set()
        time.sleep(1 / 60)
    print("# Detection mockup stopped.")
