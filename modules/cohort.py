import os
import socket
import torch
from transformers import CLIPProcessor, CLIPModel
from utils.image_loader import load_image

def _load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
    if not offline:
        try:
            socket.create_connection(("huggingface.co", 443), timeout=1)
        except OSError:
            offline = True
    kwargs.setdefault("local_files_only", offline)
    return loader.from_pretrained(model_name, **kwargs)

MODEL_NAME = "openai/clip-vit-base-patch16"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
print("Loading CLIP model...")
model = _load_hf_model(CLIPModel, MODEL_NAME).to(device)
processor = _load_hf_model(CLIPProcessor, MODEL_NAME, use_fast=False)

GENDER_PROMPTS = ["a photo, anime, cartoon, or 3D image of a boy or male character", "a photo, anime, cartoon, or 3D image of a girl or female character"]
GENDER_NAMES = ["male", "female"]
AGE_NAMES = ["child", "teenager", "young adult", "adult", "middle-aged", "elderly"]
AGE_PROMPTS = [
    f"a photo, anime, cartoon, or 3D image of a {age} adult"
    if age in ("middle-aged", "elderly")
    else f"a photo, anime, cartoon, or 3D image of a {age}"
    for age in AGE_NAMES
]
AGE_CLASSES = ["NONE", *AGE_NAMES]
AGE_SD_PROMPTS = {
    age: ("", "") if age == "NONE" else (
        f"{age} adult" if age in ("middle-aged", "elderly") else age,
        ", ".join(
            f"{other} adult" if other in ("middle-aged", "elderly") else other
            for other in AGE_NAMES
            if other != age
        ),
    )
    for age in AGE_CLASSES
}
def _predict(image_path, prompts):
    image = load_image(image_path).convert("RGB")
    inputs = processor(text=prompts, images=image, return_tensors="pt", padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        probs = outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]
    index = probs.argmax()
    return index, probs[index] * 100

def predict_gender(image_path):
    index, confidence = _predict(image_path, GENDER_PROMPTS)
    return GENDER_NAMES[index], confidence

def predict_age(image_path):
    index, confidence = _predict(image_path, AGE_PROMPTS)
    return AGE_NAMES[index], confidence