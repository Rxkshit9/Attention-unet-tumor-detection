import os
import tempfile
import shutil
import numpy as np
import nibabel as nib
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import label
from enum import Enum
from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.responses import FileResponse, JSONResponse

class ModelType(str, Enum):
    unet = "unet"
    attention = "attention"

# Create FastAPI app
app = FastAPI(
    title="3D Liver and Tumor Segmentation API",
    description="Inference API serving 3D U-Net and 3D Attention U-Net models for Liver and Tumor segmentation on LiTS dataset.",
    version="1.0.0"
)

# Global variables for models and device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
unet_model = None
attention_model = None

@app.on_event("startup")
def startup_event():
    global unet_model, attention_model
    
    workspace_dir = os.path.dirname(os.path.abspath(__file__))
    unet_path = os.path.join(workspace_dir, "unet_model.pkl")
    attention_path = os.path.join(workspace_dir, "attention_unet_model.pkl")
    
    print(f"Using device: {device}")
    
    # Load UNet3D
    if os.path.exists(unet_path):
        print(f"Loading UNet3D from {unet_path}...")
        unet_model = torch.load(unet_path, map_location=device)
        unet_model.to(device)
        unet_model.eval()
        print("UNet3D loaded successfully.")
    else:
        print(f"WARNING: UNet3D checkpoint not found at {unet_path}")
        
    # Load AttentionUNet3D
    if os.path.exists(attention_path):
        print(f"Loading AttentionUNet3D from {attention_path}...")
        attention_model = torch.load(attention_path, map_location=device)
        attention_model.to(device)
        attention_model.eval()
        print("AttentionUNet3D loaded successfully.")
    else:
        print(f"WARNING: AttentionUNet3D checkpoint not found at {attention_path}")

# ==========================================
# Preprocessing Helpers
# ==========================================
def clip_hu(volume, min_hu=-200, max_hu=250):
    return np.clip(volume, min_hu, max_hu)

def normalize(volume):
    mean = np.mean(volume)
    std = np.std(volume)
    return (volume - mean) / (std + 1e-8)

# ==========================================
# Post-processing Helpers
# ==========================================
def remove_small_regions(mask, min_size=500):
    labeled, num = label(mask)
    cleaned = np.zeros_like(mask)
    for i in range(1, num + 1):
        region = labeled == i
        if region.sum() > min_size:
            cleaned[region] = 1
    return cleaned

def largest_component(mask):
    labeled, num = label(mask)
    sizes = []
    for i in range(1, num + 1):
        sizes.append((labeled == i).sum())
    if len(sizes) == 0:
        return mask
    largest = np.argmax(sizes) + 1
    cleaned = (labeled == largest)
    return cleaned

# ==========================================
# Inference Engine
# ==========================================
def sliding_window_inference(model, volume, patch_size=(16, 128, 128), stride=(8, 64, 64)):
    model.eval()
    D, H, W = volume.shape
    d, h, w = patch_size

    output = np.zeros((3, D, H, W), dtype=np.float32)
    count = np.zeros((D, H, W), dtype=np.float32)

    for z in range(0, D - d + 1, stride[0]):
        for y in range(0, H - h + 1, stride[1]):
            for x in range(0, W - w + 1, stride[2]):
                patch = volume[z:z+d, y:y+h, x:x+w]
                patch_tensor = torch.tensor(patch).float().unsqueeze(0).unsqueeze(0).to(device)

                with torch.no_grad():
                    pred = model(patch_tensor)
                    pred = F.softmax(pred, dim=1)

                pred = pred.cpu().numpy()[0]
                output[:, z:z+d, y:y+h, x:x+w] += pred
                count[z:z+d, y:y+h, x:x+w] += 1

    output = output / np.maximum(count, 1)[None, :, :, :]
    seg = np.argmax(output, axis=0)
    return seg

# ==========================================
# REST Endpoints
# ==========================================
@app.get("/")
def health_check():
    return {
        "status": "healthy",
        "device": str(device),
        "models_loaded": {
            "unet": unet_model is not None,
            "attention": attention_model is not None
        }
    }

@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    model_type: ModelType = Form(ModelType.unet, description="Select target segmentation model: 'unet' for baseline or 'attention' for Attention U-Net."),
    min_size: int = Form(500, description="Minimum component size in voxels to retain during post-processing cleanup.")
):
    selected_model = unet_model if model_type == ModelType.unet else attention_model
    if selected_model is None:
        raise HTTPException(status_code=503, detail=f"Selected model '{model_type.value}' is not loaded on startup.")

    # 1. Save uploaded file to temp file
    temp_dir = tempfile.mkdtemp()
    input_path = os.path.join(temp_dir, file.filename)
    output_filename = "predicted_" + file.filename
    output_path = os.path.join(temp_dir, output_filename)
    
    try:
        with open(input_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
            
        # 2. Load NIfTI volume
        try:
            nii_img = nib.load(input_path)
            volume = nii_img.get_fdata()
        except Exception as e:
            raise HTTPException(status_code=422, detail=f"Failed to read NIfTI file: {str(e)}")

        # Voxel metrics
        zooms = nii_img.header.get_zooms()
        voxel_volume_mm3 = np.prod(zooms[:3])

        # 3. Preprocess
        # Transpose (H, W, D) -> (D, H, W)
        transposed_vol = np.transpose(volume, (2, 0, 1))
        clipped = clip_hu(transposed_vol)
        normalized = normalize(clipped)

        # 4. Check shape requirements
        D, H, W = normalized.shape
        if D < 16 or H < 128 or W < 128:
            raise HTTPException(
                status_code=422, 
                detail=f"Input volume dimension {normalized.shape} is too small. Minimum required: (16, 128, 128)"
            )

        # 5. Sliding Window Inference
        pred_volume = sliding_window_inference(selected_model, normalized)

        # 6. Post-processing Cleanup
        pred_liver = remove_small_regions(pred_volume == 1, min_size=min_size)
        pred_tumor = remove_small_regions(pred_volume == 2, min_size=min_size)

        # Retain only largest component for liver
        pred_liver = largest_component(pred_liver)

        # Enforce anatomical constraint: tumors must be inside the liver
        pred_tumor = pred_tumor & pred_liver

        # Re-assemble mask
        final_mask = np.zeros_like(pred_volume, dtype=np.uint8)
        final_mask[pred_liver] = 1
        final_mask[pred_tumor] = 2

        # 7. Voxel calculations
        liver_voxels = int(np.sum(final_mask == 1))
        tumor_voxels = int(np.sum(final_mask == 2))
        liver_vol_mm3 = float(liver_voxels * voxel_volume_mm3)
        tumor_vol_mm3 = float(tumor_voxels * voxel_volume_mm3)

        # 8. Transpose prediction back to (H, W, D)
        output_mask = np.transpose(final_mask, (1, 2, 0))

        # Save output NIfTI
        pred_nii = nib.Nifti1Image(output_mask, nii_img.affine, nii_img.header)
        nib.save(pred_nii, output_path)

        # Prepare response headers
        headers = {
            "X-Liver-Voxel-Count": str(liver_voxels),
            "X-Tumor-Voxel-Count": str(tumor_voxels),
            "X-Liver-Volume-mm3": f"{liver_vol_mm3:.2f}",
            "X-Tumor-Volume-mm3": f"{tumor_vol_mm3:.2f}",
            "X-Voxel-Spacing-mm": ",".join(map(str, zooms[:3])),
            "Access-Control-Expose-Headers": "X-Liver-Voxel-Count, X-Tumor-Voxel-Count, X-Liver-Volume-mm3, X-Tumor-Volume-mm3, X-Voxel-Spacing-mm"
        }

        # Return file response
        return FileResponse(
            path=output_path,
            filename=output_filename,
            media_type="application/octet-stream",
            headers=headers
        )
        
    except Exception as e:
        # Ensure cleanup in case of error
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
        raise e
