import os
import socket

import torch
from torch.nn.functional import normalize
from transformers import CLIPModel, CLIPProcessor

from utils.image_loader import load_image

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
GENDER_PROMPTS = [f'a {IMAGE_PROMPT} of a boy or male character', f'a {IMAGE_PROMPT} of a girl or female character']

CHILD = 'child'
TEEN = 'teenager'
YOUTH = 'young adult'
ADULT = 'adult'
MIDLIFE = 'middle-aged'
OLD = 'elderly'
AGE_CLASS = [CHILD, TEEN, YOUTH, ADULT, MIDLIFE, OLD]
AGE_PROMPTS = [f'a {IMAGE_PROMPT} of a {age} adult' if age in (MIDLIFE, OLD) else f'a {IMAGE_PROMPT} of a {age}' for age in AGE_CLASS]
CLASS_PROMPTS = {
    REAL: ['a real photograph', 'a real-life scene', 'a camera photo of a real person'],
    ANIME: ['anime artwork', 'Japanese anime illustration', 'cartoon anime style'],
    THREE_D: ['3D rendered CGI image', 'computer generated 3D render', '3D game character render'],
    CARTOON: ['cartoon illustration', '2D cartoon drawing', 'comic style cartoon art'],
}

MODEL_NAME = 'openai/clip-vit-base-patch16'


def _load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get('HF_HUB_OFFLINE', '').lower() in {'1', 'true', 'yes', 'on'}
    if not offline:
        try:
            socket.create_connection(('huggingface.co', 443), timeout=1).close()
        except OSError:
            offline = True
    kwargs.setdefault('local_files_only', offline)
    return loader.from_pretrained(model_name, **kwargs)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')
print('Loading CLIP model...')
processor = _load_hf_model(CLIPProcessor, MODEL_NAME, use_fast=False)
model = _load_hf_model(CLIPModel, MODEL_NAME).to(device)
model.eval()


def _encode_text(prompts):
    inputs = processor(text=prompts, return_tensors='pt', padding=True).to(device)
    with torch.no_grad():
        return normalize(model.get_text_features(**inputs), dim=-1)


CLASS_EMBEDDINGS = torch.stack([normalize(_encode_text(CLASS_PROMPTS[name]).mean(dim=0), dim=0) for name in CLASS_NAMES])
GENDER_EMBEDDINGS = _encode_text(GENDER_PROMPTS)
AGE_EMBEDDINGS = _encode_text(AGE_PROMPTS)


def _predict_scores(image_path, embeddings):
    inputs = processor(images=load_image(image_path).convert('RGB'), return_tensors='pt').to(device)
    with torch.no_grad():
        features = normalize(model.get_image_features(**inputs), dim=-1)
        return (model.logit_scale.exp() * features @ embeddings.T).softmax(dim=-1).cpu().numpy()[0]


def _predict(image_path, embeddings):
    probabilities = _predict_scores(image_path, embeddings)
    index = int(probabilities.argmax())
    return index, float(probabilities[index]) * 100


def classify_image(file_path):
    probabilities = _predict_scores(file_path, CLASS_EMBEDDINGS)
    results = {name: float(probability) for name, probability in zip(CLASS_NAMES, probabilities)}
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    best_class, best_probability = sorted_results[0]
    return {'best_class': best_class, 'best_probability': best_probability, 'margin': best_probability - sorted_results[1][1], 'results': results, 'sorted_results': sorted_results, 'device': str(device)}


def predict_gender(image_path):
    index, confidence = _predict(image_path, GENDER_EMBEDDINGS)
    return GENDER_NAMES[index], confidence


def predict_age(image_path):
    index, confidence = _predict(image_path, AGE_EMBEDDINGS)
    return AGE_CLASS[index], confidence