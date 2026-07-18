# attention-unet-tumor-detection

## Project Overview
This repository implements a 3D medical image segmentation pipeline for liver and tumor detection using attention-based U-Net architectures. The work focuses on volumetric segmentation of CT scan data and includes preprocessing, data augmentation, model training, evaluation, and visualization.

## Dataset and Data Collection
- Data collection was from Kaggle.
- The dataset used in this project is the LiTS (Liver Tumor Segmentation) dataset, sourced from Kaggle.
- The notebook pipeline loads NIfTI (`.nii`) image volumes and corresponding segmentation masks.
- Raw dataset files are not included in this repository due to size and licensing, so download the dataset locally before running the notebooks.

## Notebooks
- `Datacleaning.ipynb`: Data preprocessing and patch extraction.
  - Loads CT volumes and segmentation masks.
  - Clips Hounsfield units, normalizes intensity, removes empty slices, and extracts 3D patches.
- `UNet.ipynb`: Baseline 3D U-Net implementation.
  - Builds a 3D U-Net network in PyTorch.
  - Trains and evaluates the model on liver and tumor segmentation.
- `AttentionU_net.ipynb`: 3D Attention U-Net implementation.
  - Adds attention gates for improved focus on relevant regions.
  - Trains and evaluates the attention-based model.

## Key Features
- 3D volumetric segmentation using PyTorch.
- Attention gating for liver and tumor segmentation.
- Data augmentation with flips, rotations, intensity scaling, noise, and gamma adjustment.
- Evaluation metrics including Dice score and IoU score.
- Post-processing to remove small false-positive predictions.

## Repository Structure
- `AttentionU_net.ipynb`: Main attention U-Net training and evaluation.
- `UNet.ipynb`: Baseline U-Net training and evaluation.
- `Datacleaning.ipynb`: Preprocessing and data preparation.
- `.venv/`: Local Python environment (should be excluded from GitHub).

## Installation
1. Create a Python virtual environment:
   ```bash
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. If you have a GPU, install a CUDA-compatible version of PyTorch.

## How to Use
1. Download the LiTS dataset from Kaggle and place it in a local folder.
2. Update the dataset paths in the notebooks, for example:
   - `image_dir`
   - `mask_dir`
3. Run `Datacleaning.ipynb` first to generate training patches.
4. Run `UNet.ipynb` or `AttentionU_net.ipynb` to train and evaluate the models.

## Notes
- Keep medical image data and large `.nii` files out of version control.
- This project is notebook-based and designed for experimentation.
- Rename or reorganize notebooks into Python scripts if you want a production-ready codebase.
