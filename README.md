# GroundingSeg
**This repo is not refactored yet!!!**
**MRI images will not be released, avoiding violating licences. Code for data processing is given.**

## Installation
This project is modified from mmdetection. Follow the steps:

1. Create a conda env (**Strongly suggested!**) and activate, install Python and PyTorch.
Our project is developed under Ubuntu 22.04 + Python=3.10 + torch=2.0.1 + cu118
2. Install mmdetection. Our project is modified from mmdet==3.2.0

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
