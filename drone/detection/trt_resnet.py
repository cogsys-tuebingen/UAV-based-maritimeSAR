import os
import ctypes
import time
import sys
import argparse
import pycuda.driver as cuda
# import pycuda.autoinit  # This is needed for initializing CUDA driver

import cv2
import numpy as np
from PIL import Image
import tensorrt as trt
import torch
import ctypes
import time


class HostDeviceMem(object):
    """Simple helper data class that's a little nicer to use than a 2-tuple."""

    def __init__(self, host_mem, device_mem, shape):
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
            inputs.append(HostDeviceMem(host_mem, device_mem, shape))
        else:
            # each grid has 3 anchors, each anchor generates a detection
            # output of 7 float32 values
            # assert size == grid_sizes[output_idx] * 3 * 7 * engine.max_batch_size
            shape[0] = engine.max_batch_size
            outputs.append(HostDeviceMem(host_mem, device_mem, shape))
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


class TrtResnet(object):
    def _load_engine(self):
        with open(self.model, 'rb') as f:
            engine_data = f.read()
            trt_runtime = trt.Runtime(trt.Logger(trt.Logger.INFO))
            engine = trt_runtime.deserialize_cuda_engine(engine_data)
            return engine

    def __init__(self, model, input_shape, category_num=80, cuda_ctx=None):
        """Initialize TensorRT plugins, engine and conetxt."""
        self.model = model
        self.input_shape = input_shape
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

    def detect(self, img):
        """Detect objects in the input image."""

        # Set host input to the image. The do_inference() function
        # will copy the input to the GPU before executing.
        self.inputs[0].host = np.ascontiguousarray(img)

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

        for i in range(len(trt_outputs)):
            trt_outputs[i] = trt_outputs[i].reshape(self.outputs[i].shape)

        return trt_outputs


if __name__ == "__main__":
    from torchvision import transforms
    from resnet import resnet18
    import torch

    net = TrtResnet(model='/home/nvidia/repository/camprototyping/server/trt_checkpoints/resnet18_trt_960_540.trt',
                    input_shape=(940, 540))
    resnet = resnet18(pretrained=True)

    i = torch.ones(1, 3, 960, 540)
    print(f"Torch: {resnet(i)[1].sum()};{resnet(i)[3].sum()}")

    print(f"TRT: {net.detect(i.numpy())}")
    exit()

    fxy = 0.25
    img_data = cv2.imread('2174.png')

    # img_data = cv2.cvtColor(img_data,cv2.COLOR_RGB2BGR)
    torch_img = torch.from_numpy(cv2.resize(img_data, (0, 0), fx=fxy, fy=fxy, interpolation=cv2.INTER_CUBIC)).to(
        'cpu')  # resize half
    torch_img = torch_img.permute(2, 0, 1) / 255
    normed_torch_img = (transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(torch_img)[None]).numpy()
    test = net.detect(normed_torch_img)
    print()

