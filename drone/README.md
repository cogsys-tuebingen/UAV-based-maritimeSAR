# ARM-Software prototype
This software package contains the ARM-Software, which will later run on the drone.

## Server Architecture
The server consists of four main components. It follows a modular concept, allowing us to make changes when the interface definitions are made and for a fast exchange of improved detectors. 
 * 'arm.server.py' combines all the parts.
 * 'camera_handling' implements different interfaces with cameras.
 * 'connection' handles the video stream and the control stream. (This has to be adapted to the final system, when the
   interface is fixed. In the final version an FPGA will handle video streaming.)
 * 'detection' contains the deep learning architecture. Now the saliency detection is implemented.

Further, there are:
 - 'config.py' defines some basic configuration parameters. For example the network configuration.
 - 'trt_checkpoints' contains the trained neural network.


# Installation Instructions
The following instructions are intended to install the ARM-Software prototype on an NVIDIA Jetson AGX Xavier.

## Requirements

The versions in requirements.txt might not be up to date. Consider installing different versions. Furthermore, if pip3 is not installed, install it via sudo apt install python3-pip.

The following requirements are part of the NVIDIA Jetpack (version 4.6):
* Python >= 3.6
* OpenCV
* TensorRT

Note for installing matplotlib:

sudo apt install python3-matplotlib

Install Pygobject via
sudo apt install python3-gi
or
conda install -c conda-forge pygobject (for a conda environment)

And Gstream via
sudo apt-get install gstreamer-1.0
sudo apt-get install gir1.2-gst-rtsp-server-1.0
or
conda install -c conda-forge gstreamer
conda install -c conda-forge gst-python


The required python packages are defined in the corresponding requirements.txt. These can be installed by using:
pip3 install -r {client/server}/requirements.txt

The current version supports generic USB-cameras and Allied Vision GigE Cameras and a demo mp4 video file.

## Install PyTorch
Follow the installation instructions from:
```https://elinux.org/Jetson_Zoo```

## Install torchvision
If it could not be installed via pip3 install torchvision, consider the following steps:
Go to https://github.com/pytorch/vision and change the branch to version 0.7 (using the tags). Download the zip-Folder and extract somewhere. In the folder execute 
python setup.py install
If it does not compile properly, you need some more packages, i.e.:<br>
sudo apt-get install libswscale-dev<br>
sudo apt-get install libavformat-dev<br>
sudo apt-get install libjpeg-dev<br>
sudo apt-get install libpng-dev<br>
sudo apt-get install libavcodec-dev<br>
sudo apt-get install libavcodec57


## Install VimbaPython (only necessary for an Allied Vision GigE Camera)
```
git clone https://github.com/alliedvision/VimbaPython
cd VimbaPython
```
Now remove line 60 to 63 in ```vimba/runtime_type_check.py```.
Change line 54 of ```setup.py``` from ```python_requires = '>=3.7'``` to ```python_requires = '>=3.6```    
Now install Vimba with:  ```pip install -e .```

# Configuration
Configure the software with the following steps in ```config.py```.
1. Define the local ip address, under which this network device is reachable from the ground station (LOCAL_IP). 
2. Port 5005 (METADATA_PORT), Port 9999 (CONTROL_PORT) and the RTSP ports (8554, 554) of the ARM-board should be accessible from the ground station (TCP and UDP).
3. (Optional) Set path to the "demo.mp4" (DEMO_VIDEO_PATH)

# How to run
1. Open a terminal.
2. Execute ```sh ./start_arm_server_dev.sh``` for development mode without roi predictions or ```sh ./start_arm_server_sal.sh``` for saliency prediction mode (requires TensorRT).
3. Wait a moment. The server should be online now.



