import streamlit as st
import torch
import torchvision.transforms as transforms
import numpy as np
import cv2
from PIL import Image
import tempfile
from stream_model import BetterSegNet2  # adapt path/class name


# -------------------- MODEL LOADING -------------------- #
@st.cache_resource
def load_model():
    model = BetterSegNet2()
    model.load_state_dict(torch.load("best_model_net_improve.pth", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# -------------------- TRANSFORMS -------------------- #
normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])


# -------------------- SEGMENTATION FUNCTION -------------------- #
def segment_full_image(image_path, model, patch_size=256):
    """
    Segments a full-size image by dividing it into patches and stitching predictions.
    """
    model.eval()
    model.cpu()

    image = Image.open(image_path).convert("RGB")
    image = np.array(image)
    h, w, _ = image.shape

    mask = np.zeros((h, w), dtype=np.float32)

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = image[y:y+patch_size, x:x+patch_size]

            patch_tensor = torch.from_numpy(patch).permute(2, 0, 1).float() / 255.0
            patch_tensor = normalize(patch_tensor).unsqueeze(0)

            with torch.no_grad():
                pred = model(patch_tensor)
                prob_mask = torch.sigmoid(pred).squeeze().cpu().numpy()
                bin_mask = (prob_mask > 0.5).astype(np.float32)

            mask[y:y+patch_size, x:x+patch_size] = bin_mask

    return mask


# -------------------- OVERLAY FUNCTION -------------------- #
def overlay_segmentation_and_boxes(image, mask):
    """
    Overlays segmentation mask and bounding boxes on the original image.
    """
    image_np = np.array(image)

    # Ensure the mask is in 8-bit format
    if mask.dtype != np.uint8:
        mask_uint8 = (mask * 255).astype('uint8') if mask.max() <= 1.0 else mask.astype('uint8')
    else:
        mask_uint8 = mask

    # Contours and bounding boxes
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(cnt) for cnt in contours]

    # Create colored mask
    mask_colored = np.zeros_like(image_np)
    mask_colored[mask_uint8 > 0] = [0, 0, 255]  # Red overlay

    alpha = 0.5
    overlayed = cv2.addWeighted(image_np, 1 - alpha, mask_colored, alpha, 0)

    # Draw bounding boxes
    for x, y, w, h in boxes:
        cv2.rectangle(overlayed, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green

    return overlayed


# -------------------- STREAMLIT UI -------------------- #
def main():
    st.set_page_config(page_title="Ship Detection", layout="wide")
    wallpaper = Image.open("./AirbusStreamlit.jpg").convert("RGB")
    
    st.image(wallpaper, caption=None, width=None, use_container_width=True, clamp=False, channels="RGB", output_format="auto")
    st.title("🚢 Ship Detection")

    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name

        image = Image.open(uploaded_file).convert("RGB")
        mask = segment_full_image(tmp_path, model, patch_size=256)
        output_img = overlay_segmentation_and_boxes(image, mask)

        # Convert OpenCV BGR to RGB before displaying
        output_img_rgb = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

        # Show images side-by-side
        col1, col2 = st.columns(2)

        with col1:
            st.image(image, caption="📷 Original Image", use_container_width=True)

        with col2:
            st.image(output_img_rgb, caption="📍 Detected Ships", use_container_width=True)


if __name__ == "__main__":  # Optional but helps in some contexts
    main()