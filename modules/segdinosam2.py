import os
import sys
import gc
import re
from functools import reduce
from io import BytesIO
from dataclasses import dataclass
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from transformers import AutoProcessor, AutoModelForZeroShotObjectDetection, AutoImageProcessor, AutoModelForSemanticSegmentation
from rembg import new_session, remove
from utils.model_utils import clear_gpu, load_hf_model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SAM2_REPO_DIR = os.path.join(SCRIPT_DIR, "sam2_repo")
SAM2_CKPT_NAME = "sam2.1_hiera_tiny.pt"
SAM2_CKPT_PATH = os.path.join("checkpoints", SAM2_CKPT_NAME)
SAM2_CONFIG_NAME = "sam2.1_hiera_t.yaml"
if SAM2_REPO_DIR not in sys.path:
    sys.path.insert(0, SAM2_REPO_DIR)

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

DINO_MODEL_ID = "IDEA-Research/grounding-dino-tiny"
REMBG_MODEL_ID = "u2net"
SEGFORMER_MODEL_ID = "mattmdjaga/segformer_b2_clothes"
FASHN_MODEL_ID = "fashn-ai/fashn-human-parser"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
REMBG_PROVIDERS = ["CPUExecutionProvider"]
DINO_BOX_THRESHOLD = 0.30
DINO_TEXT_THRESHOLD = 0.20
NEG_BOX_THRESHOLD = 0.18
NEG_TEXT_THRESHOLD = 0.15
MAX_POS_AREA = 0.60
MAX_NEG_AREA = 0.45
MAX_PERSON_AREA = 0.98
SUS_NEG_HEIGHT = 0.75
SUS_NEG_WIDTH = 0.30
SUS_NEG_AREA = 0.25
NEG_OVERLAP_MAX = 0.92
MIN_BOX_AREA_RATIO = 0.002
DUP_IOU_THRESHOLD = 0.55
NESTED_THRESHOLD = 0.90
BOX_MARGIN = 0.04
MAX_DETECTIONS = {"clothing": 2, "person": 8, "negative": 2}
MIN_MASK_AREA = 100
MIN_KEEP_RATIO = 0.10
REMBG_THRESHOLD = 8
REMBG_ERODE = 1
COLOR_NAMES = ["RED", "GREEN", "BLUE", "YELLOW", "MAGENTA", "CYAN", "PURPLE", "ORANGE", "SKY BLUE", "LIME"]
SEGMENT_COLORS = np.array([(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 128, 0), (0, 128, 255), (128, 255, 0)], dtype=np.uint8)
SEGFORMER_CLASS_IDS = {"background": [0], "hat": [1], "hair": [2], "sunglasses": [3], "upper-clothes": [4], "skirt": [5], "pants": [6], "dress": [7], "belt": [8], "left-shoe": [9], "right-shoe": [10], "face": [11], "left-leg": [12], "right-leg": [13], "left-arm": [14], "right-arm": [15], "bag": [16], "scarf": [17]}
FASHN_CLASS_IDS = {"background": [0], "face": [1], "hair": [2], "top": [3], "dress": [4], "skirt": [5], "pants": [6], "belt": [7], "bag": [8], "hat": [9], "scarf": [10], "glasses": [11], "arms": [12], "hands": [13], "legs": [14], "feet": [15], "torso": [16], "jewelry": [17]}
AVAILABLE_MODELS = {"dino", "rembg", "segformer_b2", "fashn"}

@dataclass
class PromptInstruction:
    model: str
    text: str

@dataclass
class SegmentationResult:
    base_masks: list
    masks: list
    labels: list
    legend: list
    colored_result: Image.Image

def thicken_mask(mask, thickness):
    result = np.asarray(mask).astype(bool).copy()
    thickness = int(thickness)
    combine = np.logical_or if thickness > 0 else np.logical_and
    for _ in range(abs(thickness)):
        height, width = result.shape
        padded = np.pad(result, 1, mode="constant", constant_values=False)
        result = reduce(combine, (padded[dy:dy + height, dx:dx + width] for dy in range(3) for dx in range(3)))
    return result.astype(bool)

def color_masks(image, masks, labels):
    result = np.asarray(image).copy()
    legend = []
    for index, (mask, label) in enumerate(zip(masks, labels)):
        pixels = np.asarray(mask).astype(bool)
        if not np.any(pixels):
            continue
        color_index = index % len(SEGMENT_COLORS)
        color = SEGMENT_COLORS[color_index]
        blended = result[pixels].astype(np.float32) * 0.50 + color.astype(np.float32) * 0.50
        result[pixels] = np.clip(blended, 0, 255).astype(np.uint8)
        legend.append((COLOR_NAMES[color_index], color.copy(), label))
    return Image.fromarray(result), legend

def _to_device(inputs, device):
    for key, value in list(inputs.items()):
        if isinstance(value, torch.Tensor):
            inputs[key] = value.to(device)
    return inputs

def _is_gpu_problem(error, device):
    text = str(error).lower()
    return device == "cuda" and ("out of memory" in text or "expected scalar type" in text or "mat1 and mat2" in text)

def _box_area(box):
    x1, y1, x2, y2 = box
    return max(0.0, x2 - x1) * max(0.0, y2 - y1)

def _intersection_area(a, b):
    return _box_area((max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])))

class RembgSegmenter:
    def __init__(self):
        self.session = None

    def load(self):
        if self.session is not None:
            return
        print()
        print("[REMBG] Loading Rembg...")
        self.session = new_session(REMBG_MODEL_ID, providers=REMBG_PROVIDERS)
        print("[REMBG] Rembg loaded.")

    def unload(self):
        self.session = None
        clear_gpu()

    def segment_foreground(self, image):
        self.load()
        image = image.convert("RGBA")
        output = remove(image, session=self.session, only_mask=True)
        if isinstance(output, Image.Image):
            mask_image = output.convert("L")
        elif isinstance(output, (bytes, bytearray, memoryview)):
            mask_image = Image.open(BytesIO(bytes(output))).convert("L")
        elif isinstance(output, np.ndarray):
            array = np.asarray(output)
            if array.ndim > 2:
                array = np.squeeze(array)
            if array.ndim != 2:
                raise TypeError(f"Unsupported rembg mask array shape: {array.shape}")
            if array.dtype != np.uint8:
                if np.issubdtype(array.dtype, np.floating) and array.max(initial=0) <= 1.0:
                    array = np.clip(array * 255.0, 0, 255)
                array = np.clip(array, 0, 255).astype(np.uint8)
            mask_image = Image.fromarray(array, "L")
        else:
            try:
                mask_image = Image.fromarray(np.asarray(output, dtype=np.uint8), "L")
            except Exception as e:
                raise TypeError(f"Unsupported rembg output type: {type(output).__name__}") from e
        if mask_image.size != image.size:
            mask_image = mask_image.resize(image.size, Image.Resampling.NEAREST)
        mask = np.asarray(mask_image, dtype=np.uint8) > REMBG_THRESHOLD
        if REMBG_ERODE > 0:
            mask = thicken_mask(mask, -REMBG_ERODE)
        return mask.astype(bool)

    def segment_background(self, image):
        return (~self.segment_foreground(image)).astype(bool)

class HFModel:
    title = ""
    short_name = ""
    model_id = ""
    processor_loader = None
    model_loader = None
    processor_message = None

    def __init__(self):
        self.processor = None
        self.model = None
        self.preferred_device = DEVICE
        self.runtime_device = DEVICE

    def _load_model(self, device):
        self.model = load_hf_model(self.model_loader, self.model_id)
        self.model = self.model.to(device)
        self.model.eval()
        self.runtime_device = device

    def _after_gpu_load(self):
        pass

    def load(self):
        if self.model is not None:
            return
        print()
        print(f"[MODEL] Loading {self.title}...")
        self.processor = load_hf_model(self.processor_loader, self.model_id, use_fast=False)
        if self.processor_message:
            print(f"[MODEL] {self.processor_message}")
        if self.preferred_device != "cuda":
            self._load_model("cpu")
            print(f"[MODEL] {self.short_name} CPU ready.")
            return
        try:
            print(f"[MODEL] Loading {self.short_name} on GPU...")
            self._load_model("cuda")
            print(f"[MODEL] {self.short_name} GPU ready.")
            self._after_gpu_load()
        except Exception as e:
            print(f"[MODEL ERROR] {self.short_name} GPU load failed.")
            print(f"[MODEL ERROR] {e}")
            self.unload()
            self.processor = load_hf_model(self.processor_loader, self.model_id, use_fast=False)
            self._load_model("cpu")
            print(f"[MODEL] {self.short_name} CPU fallback ready.")

    def unload(self):
        if self.model is not None:
            try:
                self.model = self.model.to("cpu")
            except Exception:
                pass
        self.model = None
        self.processor = None
        clear_gpu()

    def switch_to_cpu(self):
        if self.model is None:
            return
        try:
            self.model = self.model.to("cpu")
            self.model.eval()
        except Exception:
            pass
        self.runtime_device = "cpu"
        clear_gpu()

    def require_loaded(self):
        if self.model is None or self.processor is None:
            raise RuntimeError(f"{self.title} is not loaded.")

    def run(self, make_inputs):
        inputs = _to_device(make_inputs(), self.runtime_device)
        try:
            with torch.inference_mode():
                return inputs, self.model(**inputs)
        except RuntimeError as error:
            if not _is_gpu_problem(error, self.runtime_device):
                raise
            print(f"[GPU] {self.short_name} GPU issue detected.")
            print("[GPU] Switching to CPU.")
            self.switch_to_cpu()
            inputs = _to_device(make_inputs(), "cpu")
            with torch.inference_mode():
                return inputs, self.model(**inputs)

    def release_step_memory(self):
        gc.collect()
        if self.runtime_device == "cuda":
            clear_gpu()

class GroundingDINO(HFModel):
    title = "Grounding DINO"
    short_name = "DINO"
    model_id = DINO_MODEL_ID
    processor_loader = AutoProcessor
    model_loader = AutoModelForZeroShotObjectDetection
    processor_message = "  > Processor loaded."

    def _after_gpu_load(self):
        self.print_cuda_memory()

    def print_cuda_memory(self):
        if not torch.cuda.is_available():
            return
        try:
            free_bytes, total_bytes = torch.cuda.mem_get_info()
            print(f"[CUDA] Used: {(total_bytes - free_bytes) / 1024**2:.0f} MB")
            print(f"[CUDA] Free: {free_bytes / 1024**2:.0f} MB")
            print(f"[CUDA] Total: {total_bytes / 1024**2:.0f} MB")
        except Exception:
            pass

    def detect(self, image, categories, box_threshold, text_threshold, max_area_ratio, mode):
        self.require_loaded()
        if not categories:
            return [], [], []
        all_boxes = []
        all_labels = []
        all_scores = []
        max_detections = MAX_DETECTIONS.get(mode, MAX_DETECTIONS["negative"])
        image_area = image.width * image.height
        for category in categories:
            print()
            print("-" * 70)
            print(f"[QUERY] {mode.upper()}: {category}")
            inputs, outputs = self.run(lambda: self.processor(images=image, text=[[category]], return_tensors="pt"))
            try:
                results = self.processor.post_process_grounded_object_detection(outputs, inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(image.height, image.width)])
            except TypeError:
                results = self.processor.post_process_grounded_object_detection(outputs=outputs, input_ids=inputs["input_ids"], threshold=box_threshold, text_threshold=text_threshold, target_sizes=[(image.height, image.width)])
            result = results[0]
            boxes = result.get("boxes", torch.empty((0, 4))).detach().float().cpu().numpy()
            scores = result.get("scores", torch.empty(0)).detach().float().cpu().numpy()
            local = []
            for box, score in zip(boxes, scores):
                if len(box) != 4:
                    continue
                x1, y1, x2, y2 = [float(value) for value in box]
                x1 = np.clip(x1, 0, image.width - 1)
                y1 = np.clip(y1, 0, image.height - 1)
                x2 = np.clip(x2, 0, image.width - 1)
                y2 = np.clip(y2, 0, image.height - 1)
                width = x2 - x1
                height = y2 - y1
                if width <= 1 or height <= 1:
                    continue
                area_ratio = width * height / image_area
                if area_ratio > max_area_ratio:
                    print(f"[REJECT] {category}: area={area_ratio:.1%}")
                    continue
                if area_ratio < MIN_BOX_AREA_RATIO:
                    continue
                if mode == "negative":
                    person_like = height / image.height >= SUS_NEG_HEIGHT and width / image.width >= SUS_NEG_WIDTH
                    if person_like and area_ratio >= SUS_NEG_AREA:
                        print(f"[REJECT] {category}: suspicious person-sized negative box={box.astype(int)} area={area_ratio:.1%}")
                        continue
                local.append((np.array([x1, y1, x2, y2], dtype=np.float32), float(score)))
            local.sort(key=lambda item: item[1], reverse=True)
            for box, score in local[:max_detections]:
                all_boxes.append(box)
                all_labels.append(category)
                all_scores.append(score)
                print(f"[CANDIDATE] {category} score={score:.3f} box={box.astype(int)}")
            del inputs, outputs, results, result
            self.release_step_memory()
        boxes, labels, scores = self.dedupe_boxes(all_boxes, all_labels, all_scores)
        return self.remove_nested_boxes(boxes, labels, scores)

    def dedupe_boxes(self, boxes, labels, scores):
        if len(boxes) <= 1:
            return boxes, labels, scores
        kept_boxes = []
        kept_labels = []
        kept_scores = []
        for index in np.argsort(-np.asarray(scores)):
            index = int(index)
            box = boxes[index]
            label = labels[index]
            duplicate = False
            for old_box, old_label in zip(kept_boxes, kept_labels):
                if label != old_label:
                    continue
                iou = self.box_iou(box, old_box)
                if iou >= DUP_IOU_THRESHOLD:
                    duplicate = True
                    print(f"[FILTER] Duplicate removed: {label} IoU={iou:.2f}")
                    break
            if not duplicate:
                kept_boxes.append(box)
                kept_labels.append(label)
                kept_scores.append(scores[index])
        return kept_boxes, kept_labels, kept_scores

    def remove_nested_boxes(self, boxes, labels, scores):
        if len(boxes) <= 1:
            return boxes, labels, scores
        keep = [True] * len(boxes)
        areas = [_box_area(box) for box in boxes]
        for i in range(len(boxes)):
            for j in range(len(boxes)):
                if i == j or not keep[i] or not keep[j] or labels[i] != labels[j]:
                    continue
                if self.box_containment(boxes[j], boxes[i]) >= NESTED_THRESHOLD and areas[j] > areas[i]:
                    loser = i if scores[i] <= scores[j] else j
                    keep[loser] = False
                    print(f"[FILTER] Nested box removed: {labels[loser]} score={scores[loser]:.3f}")
        kept = [index for index in range(len(boxes)) if keep[index]]
        return [boxes[i] for i in kept], [labels[i] for i in kept], [scores[i] for i in kept]

    @staticmethod
    def box_iou(a, b):
        intersection = _intersection_area(a, b)
        union = _box_area(a) + _box_area(b) - intersection
        if union <= 0:
            return 0.0
        return intersection / union

    @staticmethod
    def box_containment(outer, inner):
        inner_area = _box_area(inner)
        if inner_area <= 0:
            return 0.0
        return _intersection_area(outer, inner) / inner_area

class SemanticParser(HFModel):
    processor_loader = AutoImageProcessor
    model_loader = AutoModelForSemanticSegmentation
    class_ids = {}

    def resolve_class_ids(self, category):
        normalized = str(category).strip().lower()
        if normalized not in self.class_ids:
            supported = ", ".join(self.class_ids.keys())
            raise ValueError(f"{self.short_name} category '{category}' is not supported. Supported labels: {supported}")
        return self.class_ids[normalized], normalized

    def predict(self, image, categories):
        self.require_loaded()
        masks = []
        labels = []
        inputs, outputs = self.run(lambda: self.processor(images=image, return_tensors="pt"))
        logits = F.interpolate(outputs.logits, size=(image.height, image.width), mode="bilinear", align_corners=False)
        prediction = logits.argmax(dim=1)[0].detach().cpu().numpy()
        for category in categories:
            class_ids, normalized = self.resolve_class_ids(category)
            mask = np.isin(prediction, np.asarray(class_ids, dtype=np.int64))
            if np.any(mask):
                masks.append(mask.astype(bool))
                labels.append(category)
                print(f"[MASK] {self.short_name}: {category} -> {normalized} pixels={int(np.count_nonzero(mask))}")
            else:
                print(f"[MASK] {self.short_name}: {category} produced an empty mask.")
        del inputs, outputs, logits, prediction
        self.release_step_memory()
        return masks, labels

class SegformerB2Clothes(SemanticParser):
    title = "SegFormer B2 Clothes"
    short_name = "SegFormer"
    model_id = SEGFORMER_MODEL_ID
    class_ids = SEGFORMER_CLASS_IDS

class FashnHumanParser(SemanticParser):
    title = "FASHN Human Parser"
    short_name = "FASHN"
    model_id = FASHN_MODEL_ID
    class_ids = FASHN_CLASS_IDS

class SAM2:
    def __init__(self, repo_dir=SAM2_REPO_DIR, device=DEVICE):
        self.repo_dir = repo_dir
        self.device = device
        self.model = None
        self.predictor = None

    def load(self):
        if self.model is not None:
            return
        checkpoint = os.path.join(self.repo_dir, SAM2_CKPT_PATH)
        if not os.path.isfile(checkpoint):
            raise FileNotFoundError("SAM 2 checkpoint not found:\n\n" + checkpoint)
        config = os.path.join(self.repo_dir, "sam2", "configs", "sam2.1", SAM2_CONFIG_NAME)
        if not os.path.isfile(config):
            raise FileNotFoundError("SAM 2 config not found:\n\n" + config)
        print(f"[SAM2] Loading SAM 2 ({SAM2_CKPT_NAME}, {SAM2_CONFIG_NAME})...")
        self.model = build_sam2(config, checkpoint, device=self.device)
        self.predictor = SAM2ImagePredictor(self.model)
        print(f"[SAM2] SAM 2 ({SAM2_CKPT_NAME}, {SAM2_CONFIG_NAME}) loaded.")

    def unload(self):
        self.predictor = None
        self.model = None
        clear_gpu()

    def set_image(self, image):
        if self.predictor is None:
            raise RuntimeError("SAM 2 is not loaded.")
        self.predictor.set_image(image)

    def predict(self, box):
        if self.predictor is None:
            raise RuntimeError("SAM 2 is not loaded.")
        masks, scores, _ = self.predictor.predict(box=box, multimask_output=False)
        if masks is None or len(masks) == 0:
            raise RuntimeError("SAM 2 returned no masks.")
        scores = np.asarray(scores).reshape(-1)
        best_index = int(np.argmax(scores))
        return np.squeeze(np.asarray(masks[best_index])).astype(bool), float(scores[best_index])

    @staticmethod
    def clip_mask(mask, box, shape, margin=0.0):
        height, width = shape
        x1, y1, x2, y2 = [float(value) for value in box]
        pad_x = (x2 - x1) * margin
        pad_y = (y2 - y1) * margin
        x1, y1, x2, y2 = [int(round(value)) for value in (x1 - pad_x, y1 - pad_y, x2 + pad_x, y2 + pad_y)]
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(0, min(width - 1, x2))
        y2 = max(0, min(height - 1, y2))
        result = np.zeros((height, width), dtype=bool)
        if x2 <= x1 or y2 <= y1:
            return result
        result[y1:y2 + 1, x1:x2 + 1] = mask[y1:y2 + 1, x1:x2 + 1] > 0
        return result

class SegDinoSAM2:
    def __init__(self):
        self.dino = GroundingDINO()
        self.sam2 = SAM2()
        self.rembg = RembgSegmenter()
        self.segformer_b2 = SegformerB2Clothes()
        self.fashn = FashnHumanParser()

    def load_dino(self):
        self.dino.load()

    def unload_dino(self):
        self.dino.unload()

    def load_sam2(self):
        self.sam2.load()

    def unload_sam2(self):
        self.sam2.unload()

    def model_key(self, model_name):
        model_name = str(model_name).strip().lower()
        if model_name not in AVAILABLE_MODELS:
            supported = ", ".join(AVAILABLE_MODELS)
            raise ValueError(f"Unknown model '@{model_name}:'. Supported model names: {supported}")
        return model_name

    def parse_prompt(self, text):
        if text is None:
            return []
        text = str(text).strip()
        if not text:
            return []
        text = text.replace("\n", " ")
        instructions = []
        pattern = re.compile(r"@([A-Za-z0-9_]+)\s*:\s*\[([^\]]*)\]|@([A-Za-z0-9_]+)\s*:\s*([^,@\[]+)|([^,@]+)")
        for match in pattern.finditer(text):
            grouped_model, grouped_text, single_model, single_text, default_text = match.groups()
            if grouped_model is not None:
                model = self.model_key(grouped_model)
                categories = [item.strip() for item in grouped_text.split(",") if item.strip()]
                if not categories:
                    raise ValueError(f"Empty model group for @{grouped_model}:")
                instructions.extend(PromptInstruction(model=model, text=category) for category in categories)
            elif single_model is not None:
                model = self.model_key(single_model)
                category = single_text.strip()
                if not category:
                    raise ValueError(f"Empty category for @{single_model}:")
                instructions.append(PromptInstruction(model=model, text=category))
            elif default_text is not None:
                category = default_text.strip()
                if category:
                    instructions.append(PromptInstruction(model="dino", text=category))
        return instructions

    def split_instructions(self, instructions):
        groups = {model: [] for model in AVAILABLE_MODELS}
        for instruction in instructions:
            groups[instruction.model].append(instruction.text)
        return groups

    def rembg_segment(self, image, category):
        normalized = str(category).strip().lower()
        if normalized == "person":
            return self.rembg.segment_foreground(image), "person"
        if normalized == "background":
            return self.rembg.segment_background(image), "background"
        raise ValueError(f"@rembg: only 'person' and 'background' are supported, received '{category}'.")

    def segment(self, image, positive_prompt, negative_prompt, thickness=0):
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image.")
        image = image.convert("RGB")
        image_np = np.array(image)
        shape = image_np.shape[:2]
        pos_instructions = self.parse_prompt(positive_prompt)
        positive_groups = self.split_instructions(pos_instructions)
        negative_groups = self.split_instructions(self.parse_prompt(negative_prompt))
        if not pos_instructions:
            positive_groups["dino"] = ["all"]
        positive_boxes, positive_labels = [], []
        negative_boxes, negative_labels = [], []
        positive_masks, pos_mask_labels = [], []
        negative_masks, neg_mask_labels = [], []
        if positive_groups["dino"]:
            self.dino.load()
            positive_boxes, positive_labels, _ = self.dino.detect(image=image, categories=positive_groups["dino"], box_threshold=DINO_BOX_THRESHOLD, text_threshold=DINO_TEXT_THRESHOLD, max_area_ratio=MAX_POS_AREA, mode="clothing")
            if len(positive_boxes) == 0:
                self.unload_dino()
                raise RuntimeError("No positive object was detected by Grounding DINO.")
        if negative_groups["dino"]:
            self.dino.load()
            negative_boxes, negative_labels, _ = self.dino.detect(image=image, categories=negative_groups["dino"], box_threshold=NEG_BOX_THRESHOLD, text_threshold=NEG_TEXT_THRESHOLD, max_area_ratio=MAX_NEG_AREA, mode="negative")
        if self.dino.model is not None:
            self.unload_dino()
        targets = ((positive_groups, positive_masks, pos_mask_labels), (negative_groups, negative_masks, neg_mask_labels))
        for groups, masks, labels in targets:
            if groups["rembg"]:
                self.rembg.load()
                for category in groups["rembg"]:
                    mask, label = self.rembg_segment(image, category)
                    masks.append(mask.astype(bool))
                    labels.append(label)
        for key, parser in (("segformer_b2", self.segformer_b2), ("fashn", self.fashn)):
            for groups, masks, labels in targets:
                if groups[key]:
                    parser.load()
                    found_masks, found_labels = parser.predict(image, groups[key])
                    masks.extend(found_masks)
                    labels.extend(found_labels)
        for parser in (self.segformer_b2, self.fashn):
            if parser.model is not None:
                parser.unload()
        if self.rembg.session is not None:
            self.rembg.unload()
        if len(positive_boxes) > 0 or len(negative_boxes) > 0:
            self.sam2.load()
            self.sam2.set_image(image_np)
        for box, label in zip(positive_boxes, positive_labels):
            mask = self.sam2.clip_mask(self.sam2.predict(box)[0], box, shape, BOX_MARGIN)
            if np.any(mask):
                positive_masks.append(mask.astype(bool))
                pos_mask_labels.append(label)
        positive_union = reduce(np.logical_or, positive_masks, np.zeros(shape, dtype=bool))
        positive_union_area = np.count_nonzero(positive_union)
        for box, label in zip(negative_boxes, negative_labels):
            if self.sam2.model is None:
                self.sam2.load()
                self.sam2.set_image(image_np)
            mask = self.sam2.clip_mask(self.sam2.predict(box)[0], box, shape).astype(bool)
            mask_area = np.count_nonzero(mask)
            if mask_area == 0:
                continue
            overlap_ratio = np.count_nonzero(mask & positive_union) / max(positive_union_area, 1)
            if positive_union_area > 0 and overlap_ratio >= NEG_OVERLAP_MAX and mask_area > positive_union_area:
                print(f"[REJECT] Negative mask rejected: {label} positive-overlap={overlap_ratio:.1%} mask-area={mask_area} positive-area={positive_union_area}")
                continue
            negative_masks.append(mask)
            neg_mask_labels.append(label)
        if self.sam2.model is not None:
            self.unload_sam2()
        if not positive_masks:
            raise RuntimeError("No positive masks were produced.")
        negative_union = reduce(np.logical_or, negative_masks, np.zeros(shape, dtype=bool))
        cleaned_masks = []
        cleaned_labels = []
        occupied = np.zeros(shape, dtype=bool)
        for mask, label in zip(positive_masks, pos_mask_labels):
            original_area = np.count_nonzero(mask)
            kept = mask & ~negative_union
            kept_area = np.count_nonzero(kept)
            if kept_area < MIN_MASK_AREA:
                continue
            if original_area > 0 and kept_area < int(original_area * MIN_KEEP_RATIO):
                continue
            cleaned = kept & ~occupied
            if not np.any(cleaned):
                continue
            cleaned_masks.append(cleaned.astype(bool))
            cleaned_labels.append(label)
            occupied |= cleaned
        if not cleaned_masks:
            raise RuntimeError("No usable masks remained.")
        base_masks = [mask.copy() for mask in cleaned_masks]
        adjusted_masks = [thicken_mask(mask, int(thickness)) for mask in base_masks]
        colored_result, legend = color_masks(image_np, adjusted_masks, cleaned_labels)
        return SegmentationResult(base_masks=base_masks, masks=adjusted_masks, labels=cleaned_labels, legend=legend, colored_result=colored_result)
