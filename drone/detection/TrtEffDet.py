import os
import ctypes
import time
import sys
import argparse
import pycuda.driver as cuda
import pycuda.autoinit  # This is needed for initializing CUDA driver

import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
from src.utils import BBoxTransform, ClipBoxes, Anchors
import torch
import ctypes
import time


from torchvision.ops.boxes import nms as nms_torch

clipBoxes = ClipBoxes()
regressBoxes = BBoxTransform()


def nms(dets, thresh):
    return nms_torch(dets[:, :4], dets[:, 4], thresh)


def _preprocess_effdet(img, input_shape):
    """Preprocess an image before TRT YOLO inferencing.

    # Args
        img: int8 numpy array of shape (img_h, img_w, 3)
        input_shape: a tuple of (H, W)

    # Returns
        preprocessed img: float32 numpy array of shape (3, H, W)
    """
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    height, width = img.shape[:2]
    img = img.astype(np.float32) / 255
    img[:, :, 0] = (img[:, :, 0] - 0.485) / 0.229
    img[:, :, 1] = (img[:, :, 1] - 0.456) / 0.224
    img[:, :, 2] = (img[:, :, 2] - 0.406) / 0.225

    scale = (input_shape[0]/img.shape[0],input_shape[1]/img.shape[1])
    img = cv2.resize(img, (input_shape[1], input_shape[0]))
    #assert scale[0] == scale[1]
    scale = scale[0]

    new_img = np.zeros((input_shape[0], input_shape[1], 3))
    new_img[0:input_shape[0], 0:input_shape[1]] = img

    new_img = np.transpose(new_img, (2, 0, 1))
    new_img = new_img[None, :, :, :]
    new_img = torch.Tensor(new_img)
    return new_img.data.numpy(),scale


def _postprocess_trt(output, img,nms_threshold):
    _,_,img_h, img_w = img.shape
    boxes, confs, clss = [], [], []
    scores, labels, boxes = _postprocess_efficentdet(output,img,nms_threshold)
    return scores, labels, boxes


def _postprocess_efficentdet(output,img,nms_threshold):
    with torch.no_grad():
        classification, transformed_anchors = [torch.from_numpy(out).cuda() for out in output]
        transformed_anchors = clipBoxes(transformed_anchors, img)

        scores = torch.max(classification, dim=2, keepdim=True)[0]

        scores_over_thresh = (scores > 0.05)[0, :, 0]

        if scores_over_thresh.sum() == 0:
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]

        classification = classification[:, scores_over_thresh, :]
        transformed_anchors = transformed_anchors[:, scores_over_thresh, :]
        scores = scores[:, scores_over_thresh, :]

        try:
            anchors_nms_idx = nms(torch.cat([transformed_anchors, scores], dim=2)[0, :, :], nms_threshold)
        except Exception:
            return [torch.zeros(0), torch.zeros(0), torch.zeros(0, 4)]
        nms_scores, nms_class = classification[0, anchors_nms_idx, :].max(dim=1)

        return [nms_scores, nms_class, transformed_anchors[0, anchors_nms_idx, :]]


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""
    def __init__(self, host_mem, device_mem,shape):
        self.host = host_mem
        self.device = device_mem
        self.shape = shape

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


def allocate_buffers(engine):
    """Allocates all host/device in/out buffers required for an engine."""
    inputs = []
    outputs = []
    bindings = []
    # output_idx = 0
    stream = cuda.Stream()
    for binding in engine:
        size = trt.volume(engine.get_binding_shape(binding)) * \
               engine.max_batch_size
        dtype = trt.nptype(engine.get_binding_dtype(binding))
        shape = engine.get_binding_shape(binding)
        # Allocate host and device buffers
        host_mem = cuda.pagelocked_empty(size, dtype)
        device_mem = cuda.mem_alloc(host_mem.nbytes)
        # Append the device buffer to device bindings.
        bindings.append(int(device_mem))
        # Append to the appropriate list.
        if engine.binding_is_input(binding):
            inputs.append(HostDeviceMem(host_mem, device_mem,shape))
        else:
            # each grid has 3 anchors, each anchor generates a detection
            # output of 7 float32 values
            # assert size == grid_sizes[output_idx] * 3 * 7 * engine.max_batch_size
            outputs.append(HostDeviceMem(host_mem, device_mem,shape))
            # output_idx += 1
    return inputs, outputs, bindings, stream


def do_inference(context, bindings, inputs, outputs, stream):
    """do_inference_v2 (for TensorRT 7.0+)

    This function is generalized for multiple inputs/outputs for full
    dimension networks.
    Inputs and outputs are expected to be lists of HostDeviceMem objects.
    """
    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]


class TrtEffDet(object):
    """TrtEffDet class encapsulates things needed to run TRT EffDet."""

    def _load_engine(self):
        with open(self.model, 'rb') as f:
            engine_data = f.read()
            trt_runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            engine = trt_runtime.deserialize_cuda_engine(engine_data)
            return engine

    def __init__(self, model, input_shape, category_num=80, cuda_ctx= None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = (input_shape[1], input_shape[0])
        self.category_num = category_num
        

        self.inference_fn = do_inference
        self.trt_logger = trt.Logger(trt.Logger.INFO)
        self.engine = self._load_engine()
        self.cuda_ctx = cuda_ctx
        if self.cuda_ctx:
            self.cuda_ctx.push()

        try:
            self.context = self.engine.create_execution_context()
            
            self.inputs, self.outputs, self.bindings, self.stream = \
                allocate_buffers(self.engine)
        except Exception as e:
            raise RuntimeError('fail to allocate CUDA resources') from e
        finally:
            if self.cuda_ctx:
                self.cuda_ctx.pop()

    def __del__(self):
        """Free CUDA memories."""
        del self.outputs
        del self.inputs
        del self.stream

    def detect(self, img, metadata,nms_threshold=0.5):
        """Detect objects in the input image."""
        img_resized,scale = _preprocess_effdet(img, self.input_shape)
        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(img_resized)

        if self.cuda_ctx:
            self.cuda_ctx.push()
        trt_outputs = self.inference_fn(
            context=self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)
        if self.cuda_ctx:
            self.cuda_ctx.pop()
        toc_inference = time.time()

        for i in range(len(trt_outputs)):
            trt_outputs[i]=trt_outputs[i].reshape(self.outputs[i].shape)
        tic_post = time.time()
        scores, labels, boxes = _postprocess_trt(
            trt_outputs, img_resized, nms_threshold)
        toc_post = time.time()

        return  scores, labels, boxes, scale

