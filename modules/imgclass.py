import os
import socket
from PIL import Image, ImageOps
import torch
from transformers import CLIPProcessor, CLIPModel

REAL = "realistic"
ANIME = "anime"
THREE_D = "3D"
CARTOON = "cartoon"

def _load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get("HF_HUB_OFFLINE", "").lower() in {"1", "true", "yes", "on"}
    if not offline:
        try:
            socket.create_connection(("huggingface.co", 443), timeout=1)
        except OSError:
            offline = True
    kwargs.setdefault("local_files_only", offline)
    return loader.from_pretrained(model_name, **kwargs)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_NAME = "openai/clip-vit-base-patch32"
processor = _load_hf_model(CLIPProcessor, MODEL_NAME, use_fast=False)
model = _load_hf_model(CLIPModel, MODEL_NAME)
model = model.to(device)
model.eval()

REAL_PROMPTS = ["a real photograph", "a real life photograph", "a photograph taken with a camera", "a photo of the real world", "a natural camera photograph", "a realistic photograph of a real person", "a realistic photograph of a real place", "a real world camera image", "a candid real life photo", "a photographic image of a real object", "a genuine camera photo", "a real unedited photograph", "a natural real-world photo", "a documentary photograph", "a photograph of something that physically exists", "a realistic camera picture", "a normal photograph captured in real life", "a photographic image rather than digital artwork"]
ANIME_PROMPTS = ["an anime illustration", "a Japanese anime artwork", "a Japanese anime drawing", "a 2D anime character", "an anime screenshot", "a manga style illustration", "a Japanese animation frame", "digital anime artwork", "a cel shaded anime illustration", "a typical Japanese anime image", "an anime character illustration", "a Japanese anime scene", "a frame from an anime", "anime style digital art", "a stylized anime drawing", "a modern anime illustration", "a 2D Japanese animated artwork", "a manga or anime artwork"]
THREE_D_PROMPTS = ["a 3D rendered image", "a computer generated 3D render", "a 3D CGI artwork", "a 3D modeled character", "a 3D computer animation frame", "a Blender style 3D render", "a realistic CGI render", "a digital 3D scene", "a 3D game character render", "a computer generated three dimensional image", "a CGI image", "a computer generated 3D scene", "a digitally modeled 3D object", "a 3D animation frame", "a photorealistic CGI render", "a 3D video game render", "a computer graphics render", "a digitally created three dimensional artwork"]
CARTOON_PROMPTS = ["a cartoon illustration", "a 2D cartoon drawing", "a western cartoon illustration", "a cartoon character drawing", "a colorful cartoon artwork", "a hand drawn cartoon", "a traditional cartoon animation frame", "a comic cartoon illustration", "a 2D animated cartoon frame", "a cartoon drawing that is not anime", "a western animated cartoon", "a stylized cartoon drawing", "a cartoon animation frame", "a hand illustrated cartoon character", "a comic style cartoon", "a non-anime cartoon illustration", "a traditional 2D cartoon artwork", "a cartoon scene"]
REAL_NEGATIVE_PROMPTS = ["a digital illustration", "a cartoon drawing", "an anime illustration", "a 3D render", "a computer generated image", "CGI artwork", "a digitally generated scene"]
ANIME_NEGATIVE_PROMPTS = ["a real photograph", "a 3D render", "a western cartoon", "a computer generated 3D scene", "a camera photograph", "a realistic photograph", "a real world photo"]
THREE_D_NEGATIVE_PROMPTS = ["a real photograph", "an anime illustration", "a 2D cartoon drawing", "a hand drawn illustration", "a manga drawing", "a traditional cartoon frame", "a camera photograph"]
CARTOON_NEGATIVE_PROMPTS = ["a real photograph", "an anime illustration", "a Japanese anime drawing", "a 3D render", "a CGI artwork", "a computer generated 3D scene", "a camera photograph"]

CLASS_PROMPTS = {REAL: REAL_PROMPTS, ANIME: ANIME_PROMPTS, THREE_D: THREE_D_PROMPTS, CARTOON: CARTOON_PROMPTS}
CLASS_NEGATIVE_PROMPTS = {REAL: REAL_NEGATIVE_PROMPTS, ANIME: ANIME_NEGATIVE_PROMPTS, THREE_D: THREE_D_NEGATIVE_PROMPTS, CARTOON: CARTOON_NEGATIVE_PROMPTS}
CLASS_NAMES = [REAL, ANIME, THREE_D, CARTOON]
PROMPT_TEMPLATES = ["{}", "a photo of {}", "an image of {}", "a picture of {}", "an artwork showing {}", "this is {}", "this image is {}"]
all_prompts = []
prompt_class_ids = []

for class_index, class_name in enumerate(CLASS_NAMES):
    for prompt in CLASS_PROMPTS[class_name]:
        for template in PROMPT_TEMPLATES:
            all_prompts.append(template.format(prompt))
            prompt_class_ids.append(class_index)

all_negative_prompts = []
negative_class_ids = []

for class_index, class_name in enumerate(CLASS_NAMES):
    for prompt in CLASS_NEGATIVE_PROMPTS[class_name]:
        for template in PROMPT_TEMPLATES[:5]:
            all_negative_prompts.append(template.format(prompt))
            negative_class_ids.append(class_index)

text_inputs = processor(text=all_prompts, return_tensors="pt", padding=True)
text_inputs = {key: value.to(device) for key, value in text_inputs.items()}

negative_text_inputs = processor(text=all_negative_prompts, return_tensors="pt", padding=True)
negative_text_inputs = {key: value.to(device) for key, value in negative_text_inputs.items()}

with torch.no_grad():
    text_output = model.get_text_features(**text_inputs)
    if not isinstance(text_output, torch.Tensor):
        if hasattr(text_output, "text_embeds"):
            text_output = text_output.text_embeds
        elif hasattr(text_output, "pooler_output"):
            text_output = text_output.pooler_output
        else:
            text_output = text_output[0]
    text_features = text_output / text_output.norm(dim=-1, keepdim=True)
    negative_text_output = model.get_text_features(**negative_text_inputs)
    if not isinstance(negative_text_output, torch.Tensor):
        if hasattr(negative_text_output, "text_embeds"):
            negative_text_output = negative_text_output.text_embeds
        elif hasattr(negative_text_output, "pooler_output"):
            negative_text_output = negative_text_output.pooler_output
        else:
            negative_text_output = negative_text_output[0]
    negative_text_features = negative_text_output / negative_text_output.norm(dim=-1, keepdim=True)

def get_class_score(image_features, class_index):
    positive_indices = [index for index, value in enumerate(prompt_class_ids) if value == class_index]
    negative_indices = [index for index, value in enumerate(negative_class_ids) if value == class_index]
    positive_similarities = image_features @ text_features[positive_indices].T
    negative_similarities = image_features @ negative_text_features[negative_indices].T
    positive_mean = positive_similarities.mean(dim=1)
    positive_top = torch.topk(positive_similarities, k=max(1, positive_similarities.shape[1] // 4), dim=1).values.mean(dim=1)
    negative_mean = negative_similarities.mean(dim=1)
    return positive_mean * 0.55 + positive_top * 0.45 - negative_mean * 0.35

def _get_image_views(image):
    width, height = image.size
    side = min(width, height)
    left = max((width - side) // 2, 0)
    top = max((height - side) // 2, 0)
    right = min(left + side, width)
    bottom = min(top + side, height)
    center = image.crop((left, top, right, bottom))
    return [image, ImageOps.mirror(image), center, ImageOps.mirror(center)]

def _get_image_features(images):
    inputs = processor(images=images, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        output = model.get_image_features(**inputs)
        if not isinstance(output, torch.Tensor):
            if hasattr(output, "image_embeds"):
                output = output.image_embeds
            elif hasattr(output, "pooler_output"):
                output = output.pooler_output
            else:
                output = output[0]
        output = output / output.norm(dim=-1, keepdim=True)
    return output

def classify_image(file_path):
    image = Image.open(file_path).convert("RGB")
    image_views = _get_image_views(image)
    image_features = _get_image_features(image_views)
    view_scores = torch.stack([get_class_score(image_features, class_index) for class_index in range(len(CLASS_NAMES))], dim=1)
    mean_scores = view_scores.mean(dim=0)
    strongest_view_scores = view_scores.max(dim=0).values
    class_scores = mean_scores * 0.75 + strongest_view_scores * 0.25
    class_scores = class_scores - class_scores.mean()
    probabilities = torch.softmax(class_scores * 35.0, dim=0)
    results = {class_name: probabilities[index].item() for index, class_name in enumerate(CLASS_NAMES)}
    raw_scores = {class_name: class_scores[index].item() for index, class_name in enumerate(CLASS_NAMES)}
    sorted_results = sorted(results.items(), key=lambda x: x[1], reverse=True)
    best_class, best_probability = sorted_results[0]
    margin = sorted_results[0][1] - sorted_results[1][1]
    return {"best_class": best_class, "best_probability": best_probability, "margin": margin, "results": results, "raw_scores": raw_scores, "sorted_results": sorted_results, "device": str(device)}