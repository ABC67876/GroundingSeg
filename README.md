# GroundingSeg
**This repo is not refactored yet， not ready for out-of-box usage currently!!!**

## Installation
This project is modified from mmdetection. Follow the steps:

1. Create a conda env (**Strongly suggested!**) and activate, install Python and PyTorch.
Our project is developed under Ubuntu 22.04 + Python=3.10 + torch=2.0.1 + cu118
2. Install mmdetection. Our project is modified from mmdet==3.2.0
(**Note**: we will add proper citation for mmdetection and MM-Grounding-DINO in camera-ready version or furture work, which we accidently forgot in submission to ICASSP2025. Apologize for this mistake.)

```
pip install -U openmim==0.3.9
mim install mmengine==0.10.4
mim install mmcv==2.1.0
```
3. Install GroundingSeg

```
git clone https://github.com/ABC67876/GroundingSeg.git
cd mmdetection
pip install -v -e .
# "-v" means verbose, or more output
# "-e" means installing a project in editable mode,
# thus any local modifications made to the code will take effect without reinstallation.

pip install -r requirements.txt
# Not yet filtered for the minimum version
```
4. visit the https://github.com/ABC67876/GroundingSeg/tree/main/patch to fix some issues.

## Usage
Generally, run the https://github.com/ABC67876/GroundingSeg/blob/main/mmdetection/tools/train.py and https://github.com/ABC67876/GroundingSeg/blob/main/mmdetection/tools/test.py

For each dataset, we have a config file for detection and segmentation respectively.

Data format should be COCO for detection and COCO panoptic for segmentation.

**We are currently working on this repo for better documents. This part is under progress.**

We **will** provide scripts for data generating, pretrain weights, and refactor this project for easier usage.
