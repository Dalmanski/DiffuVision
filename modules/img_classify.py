import os
import socket
import urllib.request
from pathlib import Path
import numpy as np
import torch
from PIL import ImageOps
from torch.nn.functional import normalize
from transformers import CLIPModel, CLIPProcessor
from utils.image_loader import load_image
REAL = 'photorealistic'
ANIME = 'anime'
THREE_D = '3d'
CARTOON = 'cartoon'
CLASS_NAMES = [REAL, ANIME, THREE_D, CARTOON]
image = ['NONE', *CLASS_NAMES]
MALE = 'male'
FEMALE = 'female'
gender = ['NONE', MALE, FEMALE]
GENDER_NAMES = [MALE, FEMALE]
IMAGE_PROMPT = f'{REAL}, {ANIME}, {CARTOON}, or {THREE_D} image'
GENDER_TERMS = [['boy', 'man', 'male character'], ['girl', 'woman', 'female character']]
GENDER_TEMPLATES = [f'a {IMAGE_PROMPT} of a {{term}}', 'a portrait of a {term}', 'the face of a {term}']
GENDER_PROMPTS = [[template.format(term=term) for template in GENDER_TEMPLATES for term in terms] for terms in GENDER_TERMS]
CHILD = 'child'
TEEN = 'teenager'
YOUTH = 'young adult'
ADULT = 'adult'
MIDLIFE = 'middle-aged'
OLD = 'elderly'
age = [CHILD, TEEN, YOUTH, ADULT, MIDLIFE, OLD]
AGE_PROMPTS = [f'a {IMAGE_PROMPT} of a {age_name} adult' if age_name in (MIDLIFE, OLD) else f'a {IMAGE_PROMPT} of a {age_name}' for age_name in age]
Fview = 'front view'
Tview = 'top view'
Bview = 'back view'
LSview = 'left side view'
RSview = 'right side view'
Uview = 'bottom view'
VIEW_LABELS = [Fview, Tview, Bview, LSview, RSview, Uview]
view = ['NONE', *VIEW_LABELS]
SIDE = 'side view'
VIEW_GROUPS = [Fview, Tview, Bview, SIDE, Uview]
SIDE_INDEX = VIEW_GROUPS.index(SIDE)
TOP_INDEX = VIEW_GROUPS.index(Tview)
BOTTOM_INDEX = VIEW_GROUPS.index(Uview)
FRONT_BACK = [VIEW_GROUPS.index(Fview), VIEW_GROUPS.index(Bview)]
VIEW_PHRASES = ['facing the camera', 'seen from a high angle looking down at the head and shoulders from above', 'seen from behind with their back to the camera', 'seen in profile from the side facing left with the nose and chin visible', 'seen in profile from the side facing right with the nose and chin visible', 'seen from a low angle looking up from below with the legs closest to the camera and the sky behind']
CLASS_SUBJECTS = ['person', 'human', *[f'{name} person' for name in GENDER_NAMES]]
VIEW_SUBJECTS = ['person', 'character', 'object']
CLASS_TEMPLATES = [['a real photograph of a {s}', 'a candid photo of a {s} taken with a camera', 'a photo of a real {s} with natural skin texture and a real background', 'a DSLR portrait photo of a {s}', 'a smartphone photo of a {s} posted on social media', 'a photo of a {s} outdoors in natural lighting', 'a fashion photo of a real {s} wearing clothes standing on a street', 'a screenshot of a video of a real {s}', 'a photo'], ['anime artwork of a {s}', 'an anime illustration of a {s} with flat colors and outlines', 'a hand drawn anime {s}', 'an anime screenshot'], ['a 3D rendered CGI image of a {s}', 'a 3D model of a {s} rendered in Unreal Engine', 'a Blender 3D character render of a {s}', 'a video game character render with plastic skin'], ['a cartoon illustration of a {s}', 'a 2D comic style cartoon drawing of a {s}', 'a western cartoon']]
VIEW_TEMPLATES = ['a {label} of a {s}', 'a photo of a {s} {phrase}', 'a {s} {phrase}', 'an orthographic {label} of a {s}']
MODEL_NAME = 'openai/clip-vit-base-patch16'
POSE_URL = 'https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_full/float16/latest/pose_landmarker_full.task'
POSE_PATH = Path(__file__).resolve().parent / 'pose_landmarker_full.task'
SIDE_RATIO = 1.0
LOWER_RATIO = 1.0
VERTICAL_PENALTY = 0.3

def _short(error):
    return f'{type(error).__name__}: {str(error)[:300]}'

def _patch_protobuf():
    from google.protobuf import message_factory
    if not hasattr(message_factory.MessageFactory, 'GetPrototype'):
        message_factory.MessageFactory.GetPrototype = lambda self, descriptor: message_factory.GetMessageClass(descriptor)

def _load_pose():
    try:
        _patch_protobuf()
        import mediapipe as mp
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
        if not POSE_PATH.exists():
            urllib.request.urlretrieve(POSE_URL, POSE_PATH)
    except Exception as error:
        return None, None, _short(error)
    for source in ({'model_asset_path': str(POSE_PATH)}, {'model_asset_buffer': POSE_PATH.read_bytes()}):
        try:
            options = vision.PoseLandmarkerOptions(base_options=mp_python.BaseOptions(**source), min_pose_detection_confidence=0.3, min_pose_presence_confidence=0.3)
            return mp, vision.PoseLandmarker.create_from_options(options), None
        except Exception as error:
            message = _short(error)
    return None, None, message

def _load_hf_model(loader, model_name, **kwargs):
    offline = os.environ.get('HF_HUB_OFFLINE', '').lower() in {'1', 'true', 'yes', 'on'}
    if not offline:
        try:
            socket.create_connection(('huggingface.co', 443), timeout=1).close()
        except OSError:
            offline = True
    kwargs.setdefault('local_files_only', offline)
    return loader.from_pretrained(model_name, **kwargs)
MP, POSE, POSE_ERROR = _load_pose()
print(f'Pose landmarker: {"ready" if POSE else "unavailable - " + POSE_ERROR}')
CLASS_PROMPTS = [list(dict.fromkeys(template.format(s=subject) for template in templates for subject in CLASS_SUBJECTS)) for templates in CLASS_TEMPLATES]
VIEW_PROMPTS = [[template.format(label=label, phrase=phrase, s=subject) for template in VIEW_TEMPLATES for subject in VIEW_SUBJECTS] for label, phrase in zip(VIEW_LABELS, VIEW_PHRASES)]
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

def _encode_groups(groups):
    return torch.stack([normalize(_encode_text(prompts).mean(dim=0), dim=0) for prompts in groups])
CLASS_EMBEDDINGS = _encode_groups(CLASS_PROMPTS)
GENDER_EMBEDDINGS = _encode_groups(GENDER_PROMPTS)
AGE_EMBEDDINGS = _encode_text(AGE_PROMPTS)
VIEW_EMBEDDINGS = _encode_groups(VIEW_PROMPTS)
VIEW_GROUP_EMBEDDINGS = torch.cat([VIEW_EMBEDDINGS[:3], normalize(VIEW_EMBEDDINGS[3:5].mean(dim=0, keepdim=True), dim=-1), VIEW_EMBEDDINGS[5:]])

def _image_features(pictures):
    inputs = processor(images=pictures, return_tensors='pt').to(device)
    with torch.no_grad():
        return normalize(model.get_image_features(**inputs), dim=-1)

def _landmarks(picture):
    if POSE is None:
        return None
    result = POSE.detect(MP.Image(image_format=MP.ImageFormat.SRGB, data=np.ascontiguousarray(np.array(picture))))
    return result.pose_landmarks[0] if result.pose_landmarks else None

def _head_crop(picture):
    points = _landmarks(picture)
    if points is None:
        return None
    width, height = picture.size
    head = np.array([[point.x * width, point.y * height] for point in points[:11]])
    center = head.mean(axis=0)
    half = float(np.ptp(head, axis=0).max()) * 1.1
    if half < 16:
        return None
    return picture.crop((int(max(center[0] - half, 0)), int(max(center[1] - half * 1.2, 0)), int(min(center[0] + half, width)), int(min(center[1] + half * 0.8, height))))

def _picture(image_path, head=False):
    picture = load_image(image_path).convert('RGB')
    crop = _head_crop(picture) if head else None
    if head:
        print(f'  head crop: {"used" if crop is not None else "not found, using full image"}')
    return picture if crop is None else crop

def _predict_scores(image_path, embeddings, head=False):
    features = _image_features([_picture(image_path, head)])
    with torch.no_grad():
        return (model.logit_scale.exp() * features @ embeddings.T).softmax(dim=-1).cpu().numpy()[0]

def _faces_right(image_path):
    picture = load_image(image_path).convert('RGB')
    scores = _image_features([picture, ImageOps.mirror(picture)]) @ VIEW_EMBEDDINGS[3:5].T
    difference = scores[:, 1] - scores[:, 0]
    return bool(difference[0] > difference[1])

def _pose_view(image_path):
    if POSE is None:
        print(f'  pose: unavailable - {POSE_ERROR}')
        return None
    picture = load_image(image_path).convert('RGB')
    points = _landmarks(picture)
    if points is None:
        print('  pose: no person found')
        return None
    width, height = picture.size
    xy = lambda index: np.array([points[index].x * width, points[index].y * height])
    span = lambda first, second: float(np.linalg.norm(xy(first) - xy(second)))
    ratio = span(11, 12) / max(np.linalg.norm((xy(11) + xy(12)) / 2 - xy(0)), 1.0)
    lower = max(span(23, 24), span(25, 26)) / max(span(11, 12), 1.0)
    lower_visible = min(points[index].visibility or 0 for index in (23, 24, 25, 26)) > 0.5
    ear = max(points[7], points[8], key=lambda point: point.visibility or 0)
    print(f'  pose: shoulder/head ratio {ratio:.2f}, lower/shoulder ratio {lower:.2f}, nose {(points[0].visibility or 0):.2f}, ears {(points[7].visibility or 0):.2f}/{(points[8].visibility or 0):.2f}')
    return ratio < SIDE_RATIO, points[0].x > ear.x, (lower > LOWER_RATIO if lower_visible else None)

def _print_scores(title, image_path, labels, probabilities):
    print(f'[{title}] {image_path}')
    for label, probability in sorted(zip(labels, probabilities), key=lambda item: item[1], reverse=True):
        print(f'  {label:<18} {probability * 100:6.2f}%')

def _predict(image_path, embeddings, title, labels, head=False):
    probabilities = _predict_scores(image_path, embeddings, head)
    _print_scores(title, image_path, labels, probabilities)
    index = int(probabilities.argmax())
    return index, float(probabilities[index]) * 100

def classify_image(file_path):
    probabilities = _predict_scores(file_path, CLASS_EMBEDDINGS)
    _print_scores('class', file_path, CLASS_NAMES, probabilities)
    results = {name: float(probability) for name, probability in zip(CLASS_NAMES, probabilities)}
    sorted_results = sorted(results.items(), key=lambda item: item[1], reverse=True)
    best_class, best_probability = sorted_results[0]
    return {'best_class': best_class, 'best_probability': best_probability, 'margin': best_probability - sorted_results[1][1], 'results': results, 'sorted_results': sorted_results, 'device': str(device)}

def predict_gender(image_path):
    index, confidence = _predict(image_path, GENDER_EMBEDDINGS, 'gender', GENDER_NAMES, True)
    return GENDER_NAMES[index], confidence

def predict_age(image_path):
    index, confidence = _predict(image_path, AGE_EMBEDDINGS, 'age', age)
    return age[index], confidence

def predict_view(image_path):
    probabilities = _predict_scores(image_path, VIEW_GROUP_EMBEDDINGS)
    _print_scores('view', image_path, VIEW_GROUPS, probabilities)
    pose = _pose_view(image_path)
    adjusted = probabilities.copy()
    if pose:
        adjusted[FRONT_BACK if pose[0] else [SIDE_INDEX]] = 0
        if pose[2] is not None:
            adjusted[TOP_INDEX if pose[2] else BOTTOM_INDEX] *= VERTICAL_PENALTY
    adjusted = adjusted / adjusted.sum()
    index = int(adjusted.argmax())
    name = VIEW_GROUPS[index]
    print(f'  final group: {name}')
    if name == SIDE:
        name = RSview if (pose[1] if pose else _faces_right(image_path)) else LSview
    return name, float(adjusted[index]) * 100