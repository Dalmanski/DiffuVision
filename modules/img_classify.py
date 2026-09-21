import os
import socket

import torch
from transformers import CLIPModel, CLIPProcessor

from utils.image_loader import load_image


def _load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get('HF_HUB_OFFLINE', '').lower() in {'1', 'true', 'yes', 'on'}
    if not offline:
        try:
            socket.create_connection(('huggingface.co', 443), timeout=1)
        except OSError:
            offline = True
    kwargs.setdefault('local_files_only', offline)
    return loader.from_pretrained(model_name, **kwargs)


REAL = 'photorealistic'
ANIME = 'anime'
THREE_D = '3d'
CARTOON = 'cartoon'
CLASS_NAMES = [REAL, ANIME, THREE_D, CARTOON]
IMAGE_CLASS = ['NONE', *CLASS_NAMES]

MALE = 'male'
FEMALE = 'female'
GENDER_CLASS = ['NONE', MALE, FEMALE]
GENDER_NAMES = [MALE, FEMALE]
IMAGE_PROMPT = f'{REAL}, {ANIME}, {CARTOON}, or {THREE_D} image'
GENDER_PROMPTS = [
    f'a {IMAGE_PROMPT} of a boy or male character',
    f'a {IMAGE_PROMPT} of a girl or female character',
]

CHILD = 'child'
TEEN = 'teenager'
YOUTH = 'young adult'
ADULT = 'adult'
MIDLIFE = 'middle-aged'
OLD = 'elderly'
AGE_CLASS = [CHILD, TEEN, YOUTH, ADULT, MIDLIFE, OLD]
AGE_PROMPTS = [
    f'a {IMAGE_PROMPT} of a {age} adult'
    if age in (MIDLIFE, OLD)
    else f'a {IMAGE_PROMPT} of a {age}'
    for age in AGE_CLASS
]
CLASS_PROMPTS = {
    REAL: ['a real photograph', 'a real-life scene', 'a camera photo of a real person'],
    ANIME: ['anime artwork', 'Japanese anime illustration', 'cartoon anime style'],
    THREE_D: ['3D rendered CGI image', 'computer generated 3D render', '3D game character render'],
    CARTOON: ['cartoon illustration', '2D cartoon drawing', 'comic style cartoon art'],
}

MODEL_NAME = 'openai/clip-vit-base-patch16'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print('Loading CLIP model...')
processor = _load_hf_model(CLIPProcessor, MODEL_NAME, use_fast=False)
model = _load_hf_model(CLIPModel, MODEL_NAME).to(device)
model.eval()


def _predict_scores(image_path, prompts):
    image = load_image(image_path).convert('RGB')
    inputs = processor(text=prompts, images=image, return_tensors='pt', padding=True)
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        outputs = model(**inputs)
        return outputs.logits_per_image.softmax(dim=-1).cpu().numpy()[0]


def _predict(image_path, prompts):
    probabilities = _predict_scores(image_path, prompts)
    index = probabilities.argmax()
    return index, probabilities[index] * 100


def classify_image(file_path):
    prompts = [prompt for class_prompts in CLASS_PROMPTS.values() for prompt in class_prompts]
    probabilities = _predict_scores(file_path, prompts)

    results = {}
    offset = 0
    for class_name in CLASS_NAMES:
        class_count = len(CLASS_PROMPTS[class_name])
        results[class_name] = float(probabilities[offset:offset + class_count].mean())
        offset += class_count
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    best_class, best_probability = sorted_results[0]
    return {
        'best_class': best_class,
        'best_probability': best_probability,
        'margin': best_probability - sorted_results[1][1],
        'results': results,
        'sorted_results': sorted_results,
        'device': str(device),
    }


def predict_gender(image_path):
    index, confidence = _predict(image_path, GENDER_PROMPTS)
    return GENDER_NAMES[index], confidence


def predict_age(image_path):
    index, confidence = _predict(image_path, AGE_PROMPTS)
    return AGE_CLASS[index], confidence