import streamlit as st
import requests
import tempfile
import os
import shutil
import nibabel as nib
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Professional Page Config
st.set_page_config(
    page_title="3D Liver & Tumor Segmenter",
    page_icon="🧬",
    layout="wide"
)

# Custom premium styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.1rem;
        color: #4B5563;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #F3F4F6;
        padding: 1.2rem;
        border-radius: 0.75rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        border-left: 5px solid #2563EB;
        margin-bottom: 1rem;
    }
    .metric-title {
        font-size: 0.85rem;
        font-weight: 600;
        color: #4B5563;
        text-transform: uppercase;
        margin-bottom: 0.25rem;
    }
    .metric-value {
        font-size: 1.8rem;
        font-weight: 700;
        color: #1F2937;
    }
</style>
""", unsafe_allow_html=True)

# App Header
st.markdown('<div class="main-header">🧬 3D Liver & Tumor Segmentation Dashboard</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload a 3D medical CT scan (NIfTI format) and run deep learning inference using U-Net architectures.</div>', unsafe_allow_html=True)

# Initialize Session State variables to prevent re-running inference on slider movement
if "input_volume" not in st.session_state:
    st.session_state.input_volume = None
if "pred_volume" not in st.session_state:
    st.session_state.pred_volume = None
if "headers" not in st.session_state:
    st.session_state.headers = None
if "pred_bytes" not in st.session_state:
    st.session_state.pred_bytes = None
if "filename" not in st.session_state:
    st.session_state.filename = None

# Sidebar controls
st.sidebar.header("⚙️ Configuration")

model_option = st.sidebar.selectbox(
    "Select Segmentation Model",
    ["3D U-Net (Baseline)", "3D Attention U-Net"],
    index=0
)
model_type = "unet" if "Baseline" in model_option else "attention"

st.sidebar.markdown("---")

st.sidebar.subheader("🔍 Post-processing")
min_size = st.sidebar.slider(
    "Min Region Size (Voxels)",
    min_value=100,
    max_value=2000,
    value=500,
    step=50,
    help="Filters out false-positive segments smaller than this voxel count. Larger values reduce noise."
)

st.sidebar.markdown("---")

api_url = st.sidebar.text_input(
    "API Endpoint URL",
    value="http://127.0.0.1:8000",
    help="The base URL where the FastAPI backend server is running."
)

# File Uploader in Main Area
uploaded_file = st.file_uploader(
    "Upload 3D CT scan (.nii or .nii.gz)",
    type=["nii", "nii.gz"],
    help="Select a 3D CT scan volume from the LiTS dataset."
)

if uploaded_file is not None:
    # Check if a new file has been uploaded (reset states if so)
    if st.session_state.filename != uploaded_file.name:
        st.session_state.input_volume = None
        st.session_state.pred_volume = None
        st.session_state.headers = None
        st.session_state.pred_bytes = None
        st.session_state.filename = uploaded_file.name

    col1, col2 = st.columns([1, 2])

    with col1:
        st.info(f"📁 Loaded: `{uploaded_file.name}`")
        run_btn = st.button("🚀 Run Segmentation Inference", use_container_width=True)

    if run_btn:
        with st.spinner("Processing 3D volume... Running sliding-window inference on GPU (this may take 15-20s)..."):
            # 1. Create a temporary file of the uploaded NIfTI
            temp_dir = tempfile.mkdtemp()
            temp_input_path = os.path.join(temp_dir, uploaded_file.name)
            
            with open(temp_input_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
            
            # Read input volume for visualization
            input_nii = nib.load(temp_input_path)
            # Transpose to (D, H, W) to match API output layout internally
            st.session_state.input_volume = np.transpose(input_nii.get_fdata(), (2, 0, 1))

            # 2. Upload file to FastAPI
            files = {"file": (uploaded_file.name, open(temp_input_path, "rb"), "application/octet-stream")}
            data = {"model_type": model_type, "min_size": int(min_size)}
            
            try:
                response = requests.post(f"{api_url}/predict", files=files, data=data)
                response.raise_for_status()
                
                # Store response bytes
                st.session_state.pred_bytes = response.content
                st.session_state.headers = response.headers
                
                # 3. Read output predicted NIfTI
                temp_output_path = os.path.join(temp_dir, "prediction.nii")
                with open(temp_output_path, "wb") as f:
                    f.write(response.content)
                
                pred_nii = nib.load(temp_output_path)
                # Transpose to (D, H, W)
                st.session_state.pred_volume = np.transpose(pred_nii.get_fdata(), (2, 0, 1))
                
                st.success("🎉 Segmentation inference complete!")
                
            except Exception as e:
                st.error(f"Inference failed. Make sure the FastAPI server is running on {api_url}.\nError: {e}")
            finally:
                # Cleanup temp directory
                if os.path.exists(temp_dir):
                    shutil.rmtree(temp_dir)

    # If prediction results exist, render Dashboard and Slider
    if st.session_state.pred_volume is not None:
        headers = st.session_state.headers
        
        # Parse metrics from response headers
        liver_voxels = headers.get("X-Liver-Voxel-Count", "0")
        tumor_voxels = headers.get("X-Tumor-Voxel-Count", "0")
        liver_vol_mm3 = float(headers.get("X-Liver-Volume-mm3", "0.0"))
        tumor_vol_mm3 = float(headers.get("X-Tumor-Volume-mm3", "0.0"))
        voxel_spacing = headers.get("X-Voxel-Spacing-mm", "N/A")
        
        # Conversions (1 cc = 1000 mm³)
        liver_vol_cc = liver_vol_mm3 / 1000.0
        tumor_vol_cc = tumor_vol_mm3 / 1000.0
        
        # Display Metrics Dashboard
        st.markdown("---")
        st.subheader("📊 Segmentation Metrics Dashboard")
        
        m_col1, m_col2, m_col3 = st.columns(3)
        
        with m_col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #10B981;">
                <div class="metric-title">Predicted Liver Volume</div>
                <div class="metric-value">{liver_vol_cc:.2f} cc</div>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem;">
                    Voxel count: {int(liver_voxels):,}<br>
                    Liters: {liver_vol_cc/1000.0:.3f} L
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #EF4444;">
                <div class="metric-title">Predicted Tumor Volume</div>
                <div class="metric-value">{tumor_vol_cc:.2f} cc</div>
                <div style="font-size: 0.85rem; color: #6B7280; margin-top: 0.25rem;">
                    Voxel count: {int(tumor_voxels):,}<br>
                    Liters: {tumor_vol_cc/1000.0:.3f} L
                </div>
            </div>
            """, unsafe_allow_html=True)
            
        with m_col3:
            st.markdown(f"""
            <div class="metric-card" style="border-left-color: #3B82F6;">
                <div class="metric-title">Voxel Spacing</div>
                <div class="metric-value" style="font-size: 1.4rem; padding-top: 0.4rem; padding-bottom: 0.4rem;">{voxel_spacing} mm</div>
                <div style="font-size: 0.85rem; color: #6B7280;">
                    Volume calculation uses original spacing metadata from NIfTI file header.
                </div>
            </div>
            """, unsafe_allow_html=True)

        st.download_button(
            label="💾 Download Predicted Segmentation Mask (.nii)",
            data=st.session_state.pred_bytes,
            file_name=f"predicted_{uploaded_file.name}",
            mime="application/octet-stream",
            use_container_width=True
        )

        # 3D Slice Viewer
        st.markdown("---")
        st.subheader("🖥️ Interactive 3D Slice Viewer")
        
        num_slices = st.session_state.input_volume.shape[0]
        
        # Find index with largest liver prediction to set as default slice
        default_slice = int(np.argmax([np.sum(st.session_state.pred_volume[s] == 1) for s in range(num_slices)]))
        
        slice_idx = st.slider(
            "Select Slice index (Z-axis)",
            min_value=0,
            max_value=num_slices - 1,
            value=default_slice,
            step=1
        )
        
        # Load raw slices
        raw_img_slice = st.session_state.input_volume[slice_idx]
        mask_slice = st.session_state.pred_volume[slice_idx]
        
        # Plotting
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Left Panel: Original CT scan
        axes[0].imshow(raw_img_slice, cmap="gray")
        axes[0].set_title(f"CT Slice {slice_idx + 1} / {num_slices}", fontsize=14, fontweight="bold")
        axes[0].axis("off")
        
        # Right Panel: CT Scan + Mask Overlay
        # Custom color map: index 0: transparent, index 1: semi-transparent green (liver), index 2: semi-transparent red (tumor)
        # Using RGBA hex colors: green is #10B98188 (approx 50% opacity), red is #EF444499 (approx 60% opacity)
        custom_colors = ['#00000000', '#10B981A0', '#EF4444D0']
        cmap = ListedColormap(custom_colors)
        
        axes[1].imshow(raw_img_slice, cmap="gray")
        mask_overlay = axes[1].imshow(mask_slice, cmap=cmap, vmin=0, vmax=2)
        axes[1].set_title(f"Segmentation Overlay (Slice {slice_idx + 1})", fontsize=14, fontweight="bold")
        axes[1].axis("off")
        
        # Add legend
        # Dummy plots for legend handle creation
        import matplotlib.patches as mpatches
        liver_patch = mpatches.Patch(color='#10B981', label='Liver')
        tumor_patch = mpatches.Patch(color='#EF4444', label='Tumor')
        axes[1].legend(handles=[liver_patch, tumor_patch], loc='lower right', fontsize=12)

        st.pyplot(fig)
        plt.close(fig)
else:
    st.info("💡 Please upload a 3D NIfTI CT scan file (.nii or .nii.gz) in the file uploader above to begin.")
