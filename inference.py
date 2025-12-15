# inference.py

import argparse
import os
import sys
import webbrowser

import numpy as np
import torch
import torch.nn.functional as F

from PIL import Image, ImageGrab, ImageTk
import tkinter as tk
from tkinter import messagebox

import folium

# Import your model + clusters
from h3_classification import H3Classifier
from model_clip import CLIPModel
# from model_clip_res_unfrozen import ResCLIPModel
from clusters import get_clusters, xyz_to_latlon


# -----------------------
# CONFIG
# -----------------------

# You can adjust this to match your training config if needed.
COUNTRIES = [
    'AL', 'AD', 'AR', 'AU', 'AT', 'BD', 'BE', 'BT', 'BO', 'BW', 'BR', 'BG', 'KH', 'CA', 'CL', 'CO',
    'HR', 'CZ', 'DK', 'DO', 'EC', 'EE', 'SZ', 'FI', 'FR', 'DE', 'GH', 'GR', 'GL', 'GT', 'HU', 'IS',
    'IN', 'ID', 'IE', 'IL', 'IT', 'JP', 'JO', 'KE', 'KG', 'LV', 'LB', 'LS', 'LI', 'LT', 'LU', 'MY',
    'MX', 'MN', 'ME', 'NA', 'NL', 'NZ', 'NG', 'MK', 'NO', 'OM', 'PS', 'PA', 'PE', 'PH', 'PL', 'PT',
    'QA', 'RO', 'RU', 'RW', 'SM', 'ST', 'SN', 'RS', 'SG', 'SK', 'SI', 'ZA', 'KR', 'ES', 'LK', 'SE',
    'CH', 'TW', 'TH', 'TR', 'TN', 'UA', 'UG', 'AE', 'GB', 'US', 'UY', 'VN',
]


def build_config():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = {
        "device": device,
        "cache_dir": "cache",
        "eval_interval": 500,
        "countries": COUNTRIES,
        "num_countries": len(COUNTRIES),
        "steps": 7500,
        "max_lr_head": 1e-4,
        "batch_size": 512,
        "accum_steps": 1,
        "classes": 861,
        "tau_km": 75,
        "model": "ViT-L/14@336px",
        'backbone_unfrozen': True,
        'h3_resolution': 2,
        'h3_mappings': 'h3_utils/h3_to_class_res2_min200_ring20.json',
        'h3_counts': 'h3_utils/h3_counts_res2.json',
    }
    return config


# -----------------------
# MODEL / CLUSTERS LOADING
# -----------------------

def load_model_and_clusters(ckpt_path: str):
    """
    Load CLIPModel and cluster centers.
    """
    config = build_config()
    classifier = H3Classifier(config)
    device = config["device"]

    print(f"Using device: {device}")
    print("Loading model architecture...")
    model = CLIPModel(config).to(device)

    print(f"Loading checkpoint from: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)

    # 1. Handle if checkpoint is nested (e.g., PyTorch Lightning saves in 'state_dict')
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        raw_state = checkpoint["state_dict"]
    else:
        raw_state = checkpoint

    fixed_state = {}
    for k, v in raw_state.items():
        new_k = k
        # 2. Clean standard prefixes
        if new_k.startswith("_orig_mod."):
            new_k = new_k[len("_orig_mod."):]
        if new_k.startswith("module."):
            new_k = new_k[len("module."):]
        
        # 3. Handle 'model.' prefix mismatch
        # If your checkpoint keys look like 'model.visual...' but your model expects 'visual...'
        # (or vice-versa), this aligns them.
        # Check if the model has a 'model' attribute to decide.
        if hasattr(model, 'model') and not new_k.startswith('model.'):
             # If model wrapper has .model but checkpoint doesn't, add it (rare)
             pass 
        elif not hasattr(model, 'model') and new_k.startswith('model.'):
             # Common case: checkpoint has 'model.visual' but inference model expects 'visual'
             new_k = new_k[len("model."):]

        fixed_state[new_k] = v

    # 4. Use strict=True to ensure ALL weights are replaced. 
    # If this errors, it means the keys don't match, and you shouldn't proceed 
    # (otherwise you'd be using untuned weights).
    try:
        model.load_state_dict(fixed_state, strict=True)
        print("Success: All weights (including ViT) loaded strictly.")
    except RuntimeError as e:
        print(f"Strict loading failed. Missing/Unexpected keys:\n{e}")
        # Fallback to non-strict if you strictly only want to load what matches, 
        # but be warned this might leave ViT layers default.
        res = model.load_state_dict(fixed_state, strict=False)
        print("Loaded with strict=False. Missing keys (weights not updated):", res.missing_keys)

    model.eval()

    print("Loading cluster centers with get_clusters...")
    cluster_centers = classifier.CLASS_CENTERS_XYZ

    if cluster_centers.shape[1] < 2:
        raise ValueError(
            f"Expected cluster_centers to have at least 2 columns (lat, lon), "
            f"got shape: {cluster_centers.shape}"
        )

    print("Model and clusters loaded.")
    return model, cluster_centers, config


# -----------------------
# INFERENCE LOGIC
# -----------------------

def run_single_inference(model, cluster_centers, config, pil_image: Image.Image):
    """
    Takes a PIL image, runs it through the model, returns predicted cluster index + coordinates.
    """
    device = config["device"]

    # Use the transform from CLIPModel (same as in training)
    transform = model.eval_transform
    image = pil_image.convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(image_tensor)
        probs = F.softmax(logits, dim=1)
        pred_idx = probs.argmax(dim=1).item()
        pred_conf = probs[0, pred_idx].item()

    x, y, z = cluster_centers[pred_idx]  # assume [lat, lon, ...]
    lat, lon = xyz_to_latlon(x, y, z)
    return pred_idx, lat, lon, pred_conf


def show_map(lat: float, lon: float, cluster_idx: int, html_path: str = "prediction_map.html"):
    """
    Creates a folium map with a marker at (lat, lon) and opens it in the browser.
    """
    m = folium.Map(location=[lat, lon], zoom_start=4)
    folium.Marker(
        location=[lat, lon],
        popup=f"Cluster {cluster_idx}\n({lat:.4f}, {lon:.4f})",
    ).add_to(m)

    # Optionally draw a small circle
    folium.CircleMarker(
        location=[lat, lon],
        radius=6,
        fill=True,
        fill_opacity=0.8,
    ).add_to(m)

    m.save(html_path)
    webbrowser.open(f"file://{os.path.abspath(html_path)}")


# -----------------------
# TKINTER UI
# -----------------------

def start_gui(model, cluster_centers, config):
    device = config["device"]

    root = tk.Tk()
    root.title("GeoGuessr CLIP Inference")

    # Top info label
    info_label = tk.Label(root, text=f"Device: {device} | Press 'Paste & Predict' to use clipboard image")
    info_label.pack(padx=10, pady=5)

    # To show the pasted image (optional, small preview)
    image_label = tk.Label(root)
    image_label.pack(padx=10, pady=5)

    # Status label (cluster index, coords)
    status_var = tk.StringVar()
    status_var.set("No prediction yet.")
    status_label = tk.Label(root, textvariable=status_var)
    status_label.pack(padx=10, pady=5)

    def paste_and_predict():
        # Grab image from clipboard
        clip_obj = ImageGrab.grabclipboard()

        if clip_obj is None:
            messagebox.showerror("Error", "No image found in clipboard.")
            return

        if isinstance(clip_obj, Image.Image):
            pil_img = clip_obj
        elif isinstance(clip_obj, list) and len(clip_obj) > 0 and isinstance(clip_obj[0], Image.Image):
            pil_img = clip_obj[0]
        else:
            messagebox.showerror("Error", "Clipboard does not contain an image.")
            return

        # Optional: show a thumbnail in the GUI
        display_img = pil_img.copy()
        display_img.thumbnail((336, 336))
        tk_img = ImageTk.PhotoImage(display_img)
        image_label.configure(image=tk_img)
        image_label.image = tk_img  # keep reference

        # Run inference
        try:
            pred_idx, lat, lon, conf = run_single_inference(
                model, cluster_centers, config, pil_img
            )
        except Exception as e:
            messagebox.showerror("Error during inference", str(e))
            return

        status_var.set(
            f"Predicted cluster: {pred_idx} | "
            f"lat: {lat:.4f}, lon: {lon:.4f} | "
            f"confidence: {conf*100:.1f}%"
        )

        # Show map in browser
        try:
            show_map(lat, lon, pred_idx)
        except Exception as e:
            messagebox.showerror("Error showing map", str(e))

    paste_button = tk.Button(root, text="Paste & Predict", command=paste_and_predict)
    paste_button.pack(padx=10, pady=10)

    # Helpful keyboard shortcut: Ctrl+V to paste & predict
    def on_ctrl_v(event):
        paste_and_predict()
    root.bind_all("<Control-v>", on_ctrl_v)

    root.mainloop()


# -----------------------
# MAIN
# -----------------------

def main():
    parser = argparse.ArgumentParser(description="Inference for GeoGuessr CLIP model.")
    parser.add_argument(
        "--ckpt",
        type=str,
        default="models/neuroguessr-861-large-acw-streetview-h3-unfrozen-best.pth",
        help="Path to the .pth checkpoint (state_dict) to load.",
    )
    args = parser.parse_args()

    if not os.path.exists(args.ckpt):
        print(f"Checkpoint not found at: {args.ckpt}")
        sys.exit(1)

    model, cluster_centers, config = load_model_and_clusters(args.ckpt)
    start_gui(model, cluster_centers, config)


if __name__ == "__main__":
    main()
