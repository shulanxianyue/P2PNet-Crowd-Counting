import os
import numpy as np
import torch
import streamlit as st
from PIL import Image, ImageDraw
from p2pnet import P2PNet

# ====== Config ======
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
WEIGHT_PATH = os.path.join(BASE_DIR, "weights", "best.pth")  # <- set your weight here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
THRESHOLD = 0.5   # hard count threshold
CROP_SIZE = 224   # short side size, consistent with training/test

def _resize_short_side(img: Image.Image, target: int):
    w, h = img.size
    if min(w, h) == 0:
        return img, 1.0
    scale = target / min(w, h)
    if scale == 1.0:
        return img, 1.0
    new_w = int(round(w * scale))
    new_h = int(round(h * scale))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS), scale

def _preprocess(img: Image.Image):
    img = img.convert("RGB")
    img_np = np.array(img).astype(np.float32) / 255.0
    img_tensor = torch.from_numpy(img_np).permute(2, 0, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    img_tensor = (img_tensor - mean) / std
    return img_tensor.unsqueeze(0)


@st.cache_resource(show_spinner=False)
def _load_model(weight_path: str) -> P2PNet:
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Weights not found: {weight_path}")
    model = P2PNet().to(DEVICE)
    checkpoint = torch.load(weight_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    return model


def _draw_points(img: Image.Image, points: np.ndarray, color=(255, 0, 0), radius=2) -> Image.Image:
    draw = ImageDraw.Draw(img)
    for x, y in points:
        left_up = (x - radius, y - radius)
        right_down = (x + radius, y + radius)
        draw.ellipse([left_up, right_down], outline=color, fill=color)
    return img

def _filter_points_in_bounds(points: np.ndarray, width: int, height: int) -> np.ndarray:
    if points.size == 0:
        return points
    mask = (
        (points[:, 0] >= 0) & (points[:, 0] < width) &
        (points[:, 1] >= 0) & (points[:, 1] < height)
    )
    return points[mask]


st.set_page_config(page_title="P2PNet Demo", layout="centered")
st.title("P2PNet Crowd Counting Demo")
st.caption("Upload an image, then press Count to estimate heads and visualize predicted points.")

uploaded = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img = Image.open(uploaded)
    st.image(img, caption="Input", width="stretch")

    if st.button("Count"):
        try:
            model = _load_model(WEIGHT_PATH)
        except Exception as e:
            st.error(str(e))
        else:
            with torch.no_grad():
                resized_img, _ = _resize_short_side(img, CROP_SIZE)
                x = _preprocess(resized_img).to(DEVICE)
                outputs = model(x)
                logits = outputs["pred_logits"].squeeze(0).squeeze(-1)
                points = outputs["pred_points"].squeeze(0).cpu().numpy()
                scores = torch.sigmoid(logits).cpu().numpy()

                mask = scores > THRESHOLD
                pred_count = int(mask.sum())
                st.success(f"Count: {pred_count}")

                vis = resized_img.copy()
                w, h = vis.size
                pred_points = points[mask]
                in_bounds = _filter_points_in_bounds(pred_points, w, h)
                oob = len(pred_points) - len(in_bounds)
                if oob > 0:
                    st.warning(f"{oob} predicted points are خارج bounds and were hidden.")
                vis = _draw_points(vis, in_bounds, color=(255, 0, 0), radius=4)
                st.image(vis, caption="Predicted Points", width="stretch")
