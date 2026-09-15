import os
import socket

import torch
from transformers import CLIPProcessor, CLIPModel


def _load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
    if not offline:
        try:
            socket.create_connection(("huggingface.co", 443), timeout=1)
        except OSError:
            offline = True

    kwargs.setdefault("local_files_only", offline)
    return loader.from_pretrained(model_name, **kwargs)


MODEL_NAME = "openai/clip-vit-base-patch32"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

print("Loading CLIP model...")
model = _load_hf_model(CLIPModel, MODEL_NAME).to(device)
processor = _load_hf_model(CLIPProcessor, MODEL_NAME, use_fast=False)

PROMPTS = [
    "a photo, anime, cartoon, or 3D image of a boy or male character",
    "a photo, anime, cartoon, or 3D image of a girl or female character"
]
CLASS_NAMES = ["male", "female"]
NEUTRAL_THRESHOLD = 60.0


def predict_gender(image_path):
    from PIL import Image
    image = Image.open(image_path).convert("RGB")
    inputs = processor(text=PROMPTS, images=image, return_tensors="pt", padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]

    best_idx = probs.argmax()
    confidence = probs[best_idx] * 100

    if confidence < NEUTRAL_THRESHOLD:
        return "neutral", confidence

    return CLASS_NAMES[best_idx], confidence