# SAM3-vision-applications

Experimental implementations and practical applications using Segment Anything Model 3, a Foundation Model designed for simultaneous object detection, segmentation, and tracking by Meta, seeking computer vision workflows, robotics and innovative hardware integration

## Hardware Specs

Implementation platforms:

- Windows 11 via WSL2
- RTX 1000 Ada (6GB VRAM, 32GB RAM)
- Python 3.12, PyTorch 2.7, CUDA 12.6
- SAM 3 (3.45 GB/848M parameters)

## Features - Current and Potential

### 1. Automatic Mask Extractor
A Tool that automates the process of isolating objects from background environments, with incredible precision

- Processing: Local inference using SAM 3's unified checkpoint
- Output: Individual PNG files for each instance found, cropped to the object's bounding box

### 2. Implementation Notebooks

Jupyter Notebooks made by Meta

## Usage

- Accompanied by a Python venv source from the original Repository guidelines:

pip install torch==2.7.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126

SAM3: https://github.com/facebookresearch/sam3?tab=readme-ov-file