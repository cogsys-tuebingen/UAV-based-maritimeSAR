# Ground-Station Software

This repository contains the software for the ground station, which represents the human-machine-interface (gui). The input is a preprocessed RTPS-stream and the meta information of the region of interest. The two inputs are synced using the frame_id. Further, the region of interest are processed by an object detector for the final prediction results.

## Installation
 1. Install Python 3.9
 2. Install required packages (pip3 install -r requirements.txt)
 3. For GPU support:
	1. Install cuda toolkit 11.1
	2. Reinstall pytorch with gpu support (pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 torchaudio==0.9.0 -f https://download.pytorch.org/whl/torch_stable.html)
 4. Configuration in ```config.py```
	1. Set the ip address of the arm device (ARM_DEVICE_SYSTEM_IP and ARM_DEVICE_VIDEO_IP) 
	2. Define the gpus, which are used for the neural networks (VISIBLE_CUDA_DEVICES). Comma-separated list of 'cuda:<device_id>' entries. The command ```nvidia_smi``` lists all possible device_ids. 
	3. (Optional) Select prediction mode. (SYNC_PREDICTIONS) Possible values are True (accurate mode) and False (fast mode). This setting defines, whether the predictions will be synced with the video data.

## Start
After the installation the software can be started via:<br>
PYTHONPATH=$PYTHONPATH:. python3 main.py

## Features
* Custom region of interests can be added (two mouse clicks) and removed (X).
* The certainty threshold can be adapted dynamically. (Affects only the object predictions)
* The stream can be paused and continued.
* The ground station transmits a Meta Data packet every 100ms to the arm software.
* First version of the Continual Learning approach

