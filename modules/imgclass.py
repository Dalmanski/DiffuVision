import os
import socket
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

REAL = 'photorealistic'
ANIME = 'anime'
THREE_D = '3d'
CARTOON = 'cartoon'
CLASS_NAMES = [REAL, ANIME, THREE_D, CARTOON]

def _load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get('HF_HUB_OFFLINE', '').lower() in {'1', 'true', 'yes', 'on'}
    if not offline:
        try:
            socket.create_connection(('huggingface.co', 443), timeout=1)
        except OSError:
            offline = True
    kwargs.setdefault('local_files_only', offline)
    return loader.from_pretrained(model_name, **kwargs)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
MODEL_NAME = 'openai/clip-vit-base-patch32'
processor = _load_hf_model(CLIPProcessor, MODEL_NAME, use_fast=False)
model = _load_hf_model(CLIPModel, MODEL_NAME).to(device)
model.eval()

CLASS_PROMPTS = {
    REAL: ['a real photograph', 'a real-life scene', 'a camera photo of a real person'],
    ANIME: ['anime artwork', 'Japanese anime illustration', 'cartoon anime style'],
    THREE_D: ['3D rendered CGI image', 'computer generated 3D render', '3D game character render'],
    CARTOON: ['cartoon illustration', '2D cartoon drawing', 'comic style cartoon art'],
}

def classify_image(file_path):
    image = Image.open(file_path).convert('RGB')
    text = [prompt for prompts in CLASS_PROMPTS.values() for prompt in prompts]
    inputs = processor(text=text, images=image, return_tensors='pt', padding=True)
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits_per_image
        probs = logits.softmax(dim=-1).cpu().numpy()[0]

    results = {}
    index = 0
    for class_name in CLASS_NAMES:
        class_probs = probs[index:index + len(CLASS_PROMPTS[class_name])]
        results[class_name] = float(class_probs.mean())
        index += len(CLASS_PROMPTS[class_name])

    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    best_class, best_probability = sorted_results[0]
    margin = sorted_results[0][1] - sorted_results[1][1]
    return {
        'best_class': best_class,
        'best_probability': best_probability,
        'margin': margin,
        'results': results,
        'sorted_results': sorted_results,
        'device': str(device),
    }